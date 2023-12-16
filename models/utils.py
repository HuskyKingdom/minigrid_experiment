import numpy as np
import json
import h5py
import torch
import cv2
import torch.nn.utils.rnn as rnn_utils


from models.encoder import Encoder
from models.RSSM import RecurrentStateSpaceModel


import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class ReplayBuffer(object):
    """
    Replay buffer for training with RNN
    """
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, done):
        """
        Add experience to replay buffer
        NOTE: observation should be transformed to np.uint8 before push
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """
        Sample experiences from replay buffer (almost) uniformly
        The resulting array will be of the form (batch_size, chunk_length)
        and each batch is consecutive sequence
        NOTE: too large chunk_length for the length of episode will cause problems
        """
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index



class contrastive_dataloader:


    def __init__(self) -> None:

        # loading hd data
        with open('human_demo_data/hd_minigrid.json', 'r') as file:
            self.hd_data = json.load(file)

        self.para = get_params()


        self.state_dim = self.para["f_d"]
        self.hidden_dim = self.para["h_d"]
        self.lstm_layers = self.para["layers"]


        # loading enocder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder().to(self.device)


        
    
        self.encoder.eval()

        self.rssm = RecurrentStateSpaceModel(self.state_dim,
                                4,
                                self.hidden_dim).to(self.device)

        self.rssm.eval()
        
        self.encoder.load_state_dict(torch.load("encoder_" + self.para["rssms"] + ".pth"))
        self.rssm.load_state_dict(torch.load("rssm_" + self.para["rssms"] + ".pth"))

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

    
    def sample(self,batch_size,featur_dim):

        sampled_language = []
        sampled_states = []

        # random sample
        total_episodes = len(self.hd_data["episodes"]) - 1
        sampled_index = np.random.randint(0,high=207,size=batch_size)

        # obtain sampled actions & language & state sequence
        for index in sampled_index:

            sampled_states.append([]) # current episode
            rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

            sampled_language.append(self.hd_data["episodes"][index]["language"])
            
            # obtain encoded state sequence from sampled index
            current_episode_len = len(self.hd_data["episodes"][index]["actions"])

            for timestep in range(current_episode_len + 1): # observation one more for done
                with h5py.File('human_demo_data/' + str(index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
                    current_image = hdf5_file["imgSeq"][timestep]
                    obs = cv2.resize(current_image,(64,64),interpolation=cv2.INTER_LINEAR)
                    obs = self.preprocess_obs(obs)
                    obs = torch.as_tensor(obs, device=self.device)
                    obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
                    with torch.no_grad():
                        embedded_obs = self.encoder(obs)
                        state_posterior = self.rssm.posterior(rnn_hidden, embedded_obs)
                        state = state_posterior.sample() # s_t ~ p(s_t | h_t, o_t)

                        sampled_states[-1].append(state) # store current time_step
                        
                        if timestep == current_episode_len: # last obs
                            current_action = 3
                        else:
                            current_action = self.hd_data["episodes"][index]["actions"][timestep]
                            if current_action == 5:
                                current_action = 3
                        
                        action_emb = torch.as_tensor(self.one_hot[current_action], device=self.device).unsqueeze(0)
                        _, rnn_hidden = self.rssm.prior(state,action_emb,rnn_hidden) # update rnn hidden

                        
    

        # pack states and re-arrange sequence order
        sequence_lengths = [len(seq) for seq in sampled_states]
        sequence_tensor = torch.zeros(len(sampled_states), max(sequence_lengths),featur_dim)
        for idx, (seq, seq_len) in enumerate(zip(sampled_states, sequence_lengths)):
            seq = torch.stack(seq, dim=1).squeeze(0)
            sequence_tensor[idx, :seq_len] = seq
        sequence_lengths, perm_idx = torch.sort(torch.tensor(sequence_lengths), descending=True)
        sequence_tensor = sequence_tensor[perm_idx]

        

        packed_states = rnn_utils.pack_padded_sequence(sequence_tensor, sequence_lengths, batch_first=True)
        sampled_language = [sampled_language[i] for i in perm_idx.tolist()] # re-arranging language


        return packed_states,sampled_language,sequence_lengths.tolist()
    


    
    
        
    def det_sample(self,featur_dim,sampled_index):

        sampled_language = []
        sampled_states = []


        # obtain sampled actions & language & state sequence
        for index in sampled_index:

            sampled_states.append([]) # current episode
            rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

            sampled_language.append(self.hd_data["episodes"][index]["language"])
            
            # obtain encoded state sequence from sampled index
            current_episode_len = len(self.hd_data["episodes"][index]["actions"])

            for timestep in range(current_episode_len + 1): # observation one more for done
                with h5py.File('human_demo_data/' + str(index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
                    current_image = hdf5_file["imgSeq"][timestep]
                    obs = cv2.resize(current_image,(64,64),interpolation=cv2.INTER_LINEAR)
                    obs = self.preprocess_obs(obs)
                    obs = torch.as_tensor(obs, device=self.device)
                    obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
                    with torch.no_grad():
                        embedded_obs = self.encoder(obs)
                        state_posterior = self.rssm.posterior(rnn_hidden, embedded_obs)
                        state = state_posterior.mean # s_t ~ p(s_t | h_t, o_t)
                        sampled_states[-1].append(state) # store current time_step
                        
                        if timestep == current_episode_len: # last obs
                            current_action = 3
                        else:
                            current_action = self.hd_data["episodes"][index]["actions"][timestep]
                            if current_action == 5:
                                current_action = 3
                        
                        action_emb = torch.as_tensor(self.one_hot[current_action], device=self.device).unsqueeze(0)
                        _, rnn_hidden = self.rssm.prior(state,action_emb,rnn_hidden) # update rnn hidden

                        
    

        # pack states and re-arrange sequence order
        sequence_lengths = [len(seq) for seq in sampled_states]
        sequence_tensor = torch.zeros(len(sampled_states), max(sequence_lengths),featur_dim)
        for idx, (seq, seq_len) in enumerate(zip(sampled_states, sequence_lengths)):
            seq = torch.stack(seq, dim=1).squeeze(0)
            sequence_tensor[idx, :seq_len] = seq
        sequence_lengths, perm_idx = torch.sort(torch.tensor(sequence_lengths), descending=True)
        sequence_tensor = sequence_tensor[perm_idx]

        

        packed_states = rnn_utils.pack_padded_sequence(sequence_tensor, sequence_lengths, batch_first=True)
        sampled_language = [sampled_language[i] for i in perm_idx.tolist()] # re-arranging language


        return packed_states,sampled_language,sequence_lengths.tolist()



def one_hot_encoding(action):
    if action == 5:
        action = 3
    actions = [0,1,2,3] # 3 for 5 in real action space
    n_values = np.max(actions) + 1
    one_hot = np.eye(n_values,dtype=np.float32)[actions]

    if action == -1: # done
        return np.array([0,0,0,0])

    return one_hot[action]





def show_seq(seq,action_seq):

    episode_len = len(seq)

    fig, axes = plt.subplots(nrows=1, ncols=episode_len)

    actions = ["L","R","F","O"]

    for i in range(episode_len):


        axes[i].imshow(seq[i])

        
        if i == episode_len - 1:
            action = "Done"
        else:
            if action_seq[i] == 5:
                action = actions[3]
            else:
                action = actions[action_seq[i]]

        
        axes[i].set_title(action)
        axes[i].set_axis_off()



    plt.tight_layout()
    plt.show()

        


def restore_obs(preprocessed_obs, bit_depth=5):
    # 反归一化
    restored_obs = (preprocessed_obs + 0.5) * 2**bit_depth
    # 恢复位深度
    restored_obs *= 2**(8 - bit_depth)
    # 由于原始图像是整数，我们需要将浮点数转换回整数
    restored_obs = np.clip(restored_obs, 0, 255)  # 确保值在0到255之间
    restored_obs = restored_obs.astype(np.uint8)
    return restored_obs


def get_params():

    params = {"f_d":512,"h_d":512,"layers":5,"rssms":"10000","encoders":"compact_encoder_","num_node":256}

    return params