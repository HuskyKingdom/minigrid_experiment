# models
from models.encoder import Encoder
from models.RSSM import RecurrentStateSpaceModel
from models.utils import one_hot_encoding,contrastive_dataloader,get_params

from contrastive_models.jmf_encoder import CTR_JmfEncoder
from contrastive_models.language_encoder import CTR_LanguageEncoder


import torch
import json
import h5py
import cv2
import numpy as np

import torch.nn.functional as F

possible_actions = [0,1,2,3] 
n_values = np.max(possible_actions) + 1
one_hot = np.eye(n_values,dtype=np.float32)[possible_actions]



def preprocess_obs(obs, bit_depth=5):
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


eps_len = ["2500"]
losses = []


para = get_params()


state_dim = para["f_d"]
hidden_dim = para["h_d"]
lstm_layers = para["layers"]


for ep in eps_len:

    device = "cuda"
    test_index = 212

    data_loader = contrastive_dataloader()

    language_encoder = CTR_LanguageEncoder().to(device)
    jmf_encoder = CTR_JmfEncoder(state_dim,hidden_dim,lstm_layers).to(device)

    run_name = para["encoders"] + ep

    language_encoder.load_state_dict(torch.load("language_encoder_" + run_name + ".pth"))
    jmf_encoder.load_state_dict(torch.load("jmf_encoder_" + run_name + ".pth"))

    language_encoder.eval()
    jmf_encoder.eval()

  

    with open('human_demo_data/hd_minigrid.json', 'r') as file:
        hd_data = json.load(file)


    # test_index = test_index

    # for i in range(len(hd_data["episodes"][test_index]["actions"])+1):

    #     with h5py.File('human_demo_data/' + str(test_index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
    #         image_data = hdf5_file["imgSeq"][i] # current obs
    #         image_data = cv2.resize(image_data,(64,64),interpolation=cv2.INTER_LINEAR)

    #     obs = image_data
    #     obs = preprocess_obs(obs)
    #     obs = torch.as_tensor(obs, device=device)
    #     obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        

    #     with torch.no_grad():
    #         embedded_obs = encoder(obs)
    #         state_posterior = rssm.posterior(rnn_hidden, embedded_obs)
    #         state = state_posterior.sample() # s_t ~ p(s_t | h_t, o_t)


    #         if img_seq == None:
    #             img_seq = state
    #         else:
    #             img_seq = torch.cat((img_seq,state),dim=0)
            
    #         if i == len(hd_data["episodes"][test_index]["actions"]): # last obs
    #             current_action = 3
    #         else:
    #             current_action = hd_data["episodes"][test_index]["actions"][i]
    #             if current_action == 5:
    #                 current_action = 3
    #         action_emb = torch.as_tensor(one_hot[current_action], device=device).unsqueeze(0)
    #         _, rnn_hidden = rssm.prior(state,action_emb,rnn_hidden) # update rnn hidden


    # img_seq = img_seq.unsqueeze(0)

    verification_set = [208,209,210,211,212]
    training_set = [100,101,102,103,104]

    sampled_states, sampled_language, effective_lengths = data_loader.det_sample(state_dim,verification_set)


    language_emb = language_encoder(sampled_language)
    state_emb = jmf_encoder(sampled_states.to(device),effective_lengths)


    real_target = hd_data["episodes"][test_index]["language"]

    languages = [real_target,"go to the blue door and go to the red key","go to blue door"]


    language_emb = F.normalize(language_emb, p=2, dim=1)
    state_emb = F.normalize(state_emb, p=2, dim=1)


    # cosin similarity score
    similarity_matrix = torch.matmul(language_emb, state_emb.T) / 0.1 # cos 



    # compute loss
    labels = torch.arange(0, language_emb.size(0)).to(language_emb.device)
    similarity_matrix_lan = torch.matmul(language_emb, state_emb.T) / 0.1 # cos 
    lan_loss = F.cross_entropy(similarity_matrix_lan, labels)

    similarity_matrix_jmf = torch.matmul(state_emb, language_emb.T) / 0.1 # cos 
    jmf_loss = F.cross_entropy(similarity_matrix_jmf, labels)

    total_loss = (lan_loss + jmf_loss) * 0.5

    losses.append(total_loss.item())

    print(similarity_matrix_lan)
    print(sampled_language)


print(losses)