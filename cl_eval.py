#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# for human demo
import json 
import matplotlib.pyplot as plt
import cv2
import h5py
import os
import torch
import numpy as np
import torch.nn.functional as F

# models
from models.encoder import Encoder
from models.RSSM import RecurrentStateSpaceModel
from models.utils import one_hot_encoding

from contrastive_models.jmf_encoder import CTR_JmfEncoder
from contrastive_models.language_encoder import CTR_LanguageEncoder

def restore_obs(preprocessed_obs, bit_depth=5):
    # 反归一化
    restored_obs = (preprocessed_obs + 0.5) * 2**bit_depth
    # 恢复位深度
    restored_obs *= 2**(8 - bit_depth)
    # 由于原始图像是整数，我们需要将浮点数转换回整数
    restored_obs = np.clip(restored_obs, 0, 255)  # 确保值在0到255之间
    restored_obs = restored_obs.astype(np.uint8)
    return restored_obs


class CL_Evaluator:
    def __init__(
        self,
        env: Env,
        eval_episodes,
        model_dirs,
        seed=None
    ) -> None:
        
        self.env = env
        self.seed = seed
        self.closed = False
        self.eval_episodes = eval_episodes
        self.episode_states_buffer = None
        self.z = 0
        

        # models 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 1024
        self.hidden_size = 512
        self.rnn_hidden = torch.zeros(1, self.hidden_size, device=self.device)

        self.language_encoder = CTR_LanguageEncoder().to(self.device)
        self.jmf_encoder = CTR_JmfEncoder(self.feature_dim,self.hidden_size,5).to(self.device)
        
        self.language_encoder.load_state_dict(torch.load(model_dirs[0]))
        self.jmf_encoder.load_state_dict(torch.load(model_dirs[1]))
        
        self.language_encoder.eval()
        self.jmf_encoder.eval()

        self.encoder = Encoder().to(self.device)
        self.rssm = RecurrentStateSpaceModel(self.feature_dim,
                                4,
                                self.hidden_size).to(self.device)
        
        self.encoder.load_state_dict(torch.load('encoder_20000.pth'))
        self.rssm.load_state_dict(torch.load('rssm_20000.pth'))
        
        self.encoder.eval()
        self.rssm.eval()


        # action one-hot
        possible_actions = [0,1,2,3] 
        n_values = np.max(possible_actions) + 1
        self.one_hot = np.eye(n_values,dtype=np.float32)[possible_actions]
        
    

    def preprocess_obs(self,obs, bit_depth=5):
        """
        Reduces the bit depth of image for the ease of training
        and convert to [-0.5, 0.5]
        In addition, add uniform random noise same as original implementation
        """
        obs = obs.astype(np.float32)
        reduced_obs = np.floor(obs / 2 ** (8 - bit_depth))
        normalized_obs = reduced_obs / 2**bit_depth - 0.5
        normalized_obs += np.random.uniform(0.0, 1.0 / 2**bit_depth, normalized_obs.shape)
        return normalized_obs


    def start(self): # start evaluation


        for episode in range(self.eval_episodes): 

            # get init obs & language instruction
            obs = self.reset(self.seed)
            observation = obs["image"]
            target = obs["mission"]
            terminated = False


            # infer language embedding
            with torch.no_grad():
                language_emb = self.language_encoder([target])
            

            while not terminated:
                
                # model prective control
                action = self.model_infer(observation,language_emb)

                # step action
                obs, reward, terminated, truncated, _ = self.env.step(action)
                # print(f"step={self.env.step_count}, reward={reward:.2f}")
                
                observation = obs["image"]
                self.env.render()

                self.z += 1

    def imagine(self,state,rnn_hidden): # given a state, image a consecutive for next time step from 4 different actions
        
        states = None
        rnns = None
        for c_action in range(4):
            action_emb = torch.as_tensor(self.one_hot[c_action], device=self.device).unsqueeze(0)
            with torch.no_grad():
                next_state_prior, next_rnn_hidden = self.rssm.prior(state,action_emb,rnn_hidden)
                next_state = next_state_prior.sample()
                if states == None:
                    states = next_state
                    rnns = next_rnn_hidden
                else:
                    states = torch.cat((states,next_state),dim=0)
                    rnns = torch.cat((rnns,next_rnn_hidden),dim=0)

        # if self.z >= 1:
        #     from models.observation_model import ObservationModel
        #     obs_model = ObservationModel(1024, 512).to(self.device)
        #     obs_model.load_state_dict(torch.load('obs_model.pth'))
        #     obs_model.eval()
            
        #     # current

        #     infer_obs = obs_model(state,rnn_hidden).squeeze(0).transpose(0, 1).transpose(1, 2)
        #     infer_obs = infer_obs.detach().cpu().numpy()

        #     infer_obs = cv2.resize(infer_obs,(224,224))
        #     infer_obs = restore_obs(infer_obs)
        #     cv2.imshow("Predicted current Obs",infer_obs)
        #     cv2.waitKey(0)

        #     # next
        #     for i in range (4):
        #         infer_obs = obs_model(states[i].unsqueeze(0),rnn_hidden).squeeze(0).transpose(0, 1).transpose(1, 2)
        #         infer_obs = infer_obs.detach().cpu().numpy()

        #         infer_obs = cv2.resize(infer_obs,(224,224))
        #         infer_obs = restore_obs(infer_obs)
        #         cv2.imshow("Predicted Next Obs " + str(i),infer_obs)
        #         cv2.waitKey(0)


        return (states,rnns)

    def soft_max(self,arr):
        e_x = np.exp(arr - np.max(arr)) 
        return e_x / e_x.sum()

    def model_infer(self,observation,language_emb):
        
        # imaging consecutive 2 steps, with 4 actions, resulting 4^2 = 16 sequences
        observation = cv2.resize(observation,(64,64),interpolation=cv2.INTER_LINEAR)
        observation = self.preprocess_obs(observation)
        observation = torch.as_tensor(observation, device=self.device)
        observation = observation.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        
        with torch.no_grad():

            embedded_obs = self.encoder(observation)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample() # s_t ~ p(s_t | h_t, o_t)
            
            # store current hidden state
            if self.episode_states_buffer == None: # start of an episode
                self.episode_states_buffer = state # (1,1024)
            else:
                self.episode_states_buffer = torch.cat((self.episode_states_buffer,state),dim=0)
            
            # imaging 16 sequeces for every actions, first 16 with first action 0..
            # in shape (16,seq_len,1024)
            imagined_seq = torch.zeros(16,2,self.feature_dim,device=self.device)
            self.z = 1
            output = self.imagine(state,self.rnn_hidden)
            self.z = 0
            consc_seqs1 = output[0]
            consc_rnns1 = output[1]
            idx = 0
            for seq in range(consc_seqs1.shape[0]):

                n_output = self.imagine(consc_seqs1[seq].unsqueeze(0),consc_rnns1[seq].unsqueeze(0))
                n_output = n_output[0]

                for n_seq in range(n_output.shape[0]):
                    imagined_seq[idx] = torch.cat((consc_seqs1[seq].unsqueeze(0),n_output[n_seq].unsqueeze(0)),dim=0)
                    idx += 1

            # making whole seq = real_seq + imgined seq
            real_seq = self.episode_states_buffer.unsqueeze(0).repeat(16, 1, 1) # duplicate real tensors
            final_seq = torch.cat((real_seq,imagined_seq),dim=1)



            # infer contrastive embedding
            cl_state_embeddings = self.jmf_encoder(final_seq,None)

            

            cl_language_embedding = F.normalize(language_emb, p=2, dim=1)
            cl_state_embeddings = F.normalize(cl_state_embeddings, p=2, dim=1)


            # cosin similarity score
            similarity_matrix = torch.matmul(cl_language_embedding, cl_state_embeddings.T)
            similarity_matrix = similarity_matrix[0]

            
            # calculate accumulated score of each action
            similarity_matrix = similarity_matrix.cpu()
            score_matrix = similarity_matrix.numpy()
            accumulated_action_score = score_matrix.reshape((4, 4)).sum(axis=1)

            accumulated_action_score = self.soft_max(accumulated_action_score)

            print(accumulated_action_score)
            
            # sampled_action = np.random.choice(np.arange(len(accumulated_action_score)), p=accumulated_action_score)
            sampled_action =  np.argmax(accumulated_action_score)

        
        
        
        # update rnn hidden
        c_action = torch.as_tensor(self.one_hot[sampled_action], device=self.device).unsqueeze(0)
        _, self.rnn_hidden = self.rssm.prior(state,c_action,self.rnn_hidden)
            
        
        if sampled_action == 3:
            sampled_action = 5


        return sampled_action

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.c_obs.append(obs["image"])
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.c_obs.append(obs["image"])
            self.env.render()

    def reset(self, seed=None):
        
        self.episode_states_buffer = None
        self.rnn_hidden = torch.zeros(1, self.hidden_size, device=self.device)
        obs,info = self.env.reset(seed=seed)
        self.env.render()

        return obs # obs-image obs-mission




if __name__ == "__main__":

    env: MiniGridEnv = gym.make(
        "BabyAI-GoToSeqS5R2-v0",
        tile_size=32,
        render_mode="human",
        agent_pov=False,
        agent_view_size=7,
        screen_size=640,
    )

    evaluate_episodes = 5000
    models = ("language_encoder_new_encoders_8000.pth","jmf_encoder_new_encoders_8000.pth")

    env = RGBImgPartialObsWrapper(env, 32)
    

    manual_control = CL_Evaluator(env,evaluate_episodes,models)
    manual_control.start()
