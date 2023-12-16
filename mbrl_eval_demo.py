import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
import numpy as np
import json
import h5py
import cv2


from models.encoder import Encoder
from models.reward_model import RewardModel
from models.observation_model import ObservationModel
from models.RSSM import RecurrentStateSpaceModel
from models.utils import one_hot_encoding,get_params

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def restore_obs(preprocessed_obs, bit_depth=5):
    # 反归一化
    restored_obs = (preprocessed_obs + 0.5) * 2**bit_depth
    # 恢复位深度
    restored_obs *= 2**(8 - bit_depth)
    # 由于原始图像是整数，我们需要将浮点数转换回整数
    restored_obs = np.clip(restored_obs, 0, 255)  # 确保值在0到255之间
    restored_obs = restored_obs.astype(np.uint8)
    return restored_obs



evaluation_index = 219

para = get_params()


state_dim = para["f_d"]
hidden_dim = para["h_d"]



# define models and loads
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().to(device)
rssm = RecurrentStateSpaceModel(state_dim,
                                4,
                                hidden_dim).to(device)
obs_model = ObservationModel(state_dim, hidden_dim).to(device)
encoder.load_state_dict(torch.load("encoder_" + para["rssms"] + ".pth"))
rssm.load_state_dict(torch.load("rssm_" + para["rssms"] + ".pth"))
obs_model.load_state_dict(torch.load("obs_model_" + para["rssms"] + ".pth"))
encoder.eval()
rssm.eval()
obs_model.eval()
rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=device)



with open('human_demo_data/hd_minigrid.json', 'r') as file:
    hd_data = json.load(file)



episode_len = len(hd_data["episodes"][evaluation_index]["actions"]) + 1


fig, axes = plt.subplots(nrows=2, ncols=episode_len)


actions = ["L","R","F","O"]

# ground truth
for i in range(episode_len):

    with h5py.File('human_demo_data/' + str(evaluation_index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
        image_data = hdf5_file["imgSeq"][i] # current obs
        image_data = cv2.resize(image_data,(64,64),interpolation=cv2.INTER_LINEAR)
    
    if i == episode_len - 1:
        action = "Done"
    else:
        if hd_data["episodes"][evaluation_index]["actions"][i] == 5:
            action = actions[3]
        else:
            action = actions[hd_data["episodes"][evaluation_index]["actions"][i]]

    axes[0][i].imshow(image_data)
    axes[0][i].set_title(action)
    axes[0][i].set_axis_off()


# predicted

with h5py.File('human_demo_data/' + str(evaluation_index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
    image_data = hdf5_file["imgSeq"][0] # current obs
    image_data = cv2.resize(image_data,(64,64),interpolation=cv2.INTER_LINEAR)

    obs = image_data

    obs = preprocess_obs(obs)
    obs = torch.as_tensor(obs, device=device)
    obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

    embedded_obs = encoder(obs)
    state_posterior = rssm.posterior(rnn_hidden, embedded_obs)
    state = state_posterior.mean # s_t ~ p(s_t | h_t, o_t)

    

    # predicted obs for t given s_t h_t -> p(o_t | s_t, h_t)
    infer_obs = obs_model(state,rnn_hidden).squeeze(0).transpose(0, 1).transpose(1, 2)
    infer_obs = infer_obs.detach().cpu().numpy()
    infer_obs = restore_obs(infer_obs)


    # show images
    axes[1][0].imshow(infer_obs)
    axes[1][0].set_title(actions[hd_data["episodes"][evaluation_index]["actions"][0]])
    axes[1][0].set_axis_off()


    # updating rnn hidden
    c_action = one_hot_encoding(hd_data["episodes"][evaluation_index]["actions"][0])
    c_action = torch.as_tensor(c_action, device=device).unsqueeze(0)

    # update rnn hidden and predict next state prior h_t+1 = f(h_t, s_t, a_t) , p(s_t+1 | h_t+1)
    state_prior, rnn_hidden = rssm.prior(state,c_action,rnn_hidden) # s_t+1 ~ p(s_t+1 | h_t+1) , h_t+1 = f(h_t, s_t, a_t) update

    



for i in range(1,episode_len):

    state = state_prior.sample()

    if i == episode_len -1 :
        c_action = one_hot_encoding(3)
        c_action = torch.as_tensor(c_action, device=device).unsqueeze(0)
    else:
        c_action = one_hot_encoding(hd_data["episodes"][evaluation_index]["actions"][i])
        c_action = torch.as_tensor(c_action, device=device).unsqueeze(0)


    infer_obs = obs_model(state,rnn_hidden).squeeze(0).transpose(0, 1).transpose(1, 2)
    infer_obs = infer_obs.detach().cpu().numpy()
    infer_obs = restore_obs(infer_obs)

    state_prior, rnn_hidden  = rssm.prior(state,c_action,rnn_hidden) # s_t+1 ~ p(s_t+1 | h_t+1) , h_t+1 = f(h_t, s_t, a_t) update

    axes[1][i].imshow(infer_obs)

    
    if i == episode_len - 1:
        action = "Done"
    else:
        if hd_data["episodes"][evaluation_index]["actions"][i] == 5:
            action = actions[3]
        else:
            action = actions[hd_data["episodes"][evaluation_index]["actions"][i]]

    
    axes[1][i].set_title(action)
    axes[1][i].set_axis_off()






plt.tight_layout()

# 显示图形
plt.show()

    
   



    
