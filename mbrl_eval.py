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


end_flg = 1
c_time = 0

while end_flg == 1:

    fig, axes = plt.subplots(2, 3, figsize=(15, 5))

    imgs = []
    captions = ["Current True Obs","Current Posterior", "Next by action Left","Next by action Right","Next by action Forward","Next by action Open"]

    # ground truth
    with h5py.File('human_demo_data/' + str(evaluation_index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
        image_data = hdf5_file["imgSeq"][c_time] # current obs
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


    # show images
    imgs.append(image_data)
    image_data = cv2.resize(image_data,(224,224))
    # cv2.imshow("True",image_data)
    # cv2.waitKey(0)
    

    infer_obs = cv2.resize(infer_obs,(224,224))
    infer_obs = restore_obs(infer_obs)
    imgs.append(infer_obs)
    # cv2.imshow("Predicted Current Obs",infer_obs)
    # cv2.waitKey(0)
    


    # imaging

    for i in range(4): 

        i_action = one_hot_encoding(i)
        i_action = torch.as_tensor(i_action, device=device).unsqueeze(0)
        
        next_state_prior, _ = rssm.prior(state,i_action,rnn_hidden)

        next_state = next_state_prior.sample()

        # predicted obs for t+1 given s_t+1 h_t+1 -> p(o_t+1 | s_t+1, h_t+1)
        next_infer_obs = obs_model(next_state,rnn_hidden).squeeze(0).transpose(0, 1).transpose(1, 2)
        next_infer_obs = next_infer_obs.detach().cpu().numpy()

        next_infer_obs = cv2.resize(next_infer_obs,(224,224))
        next_infer_obs = restore_obs(next_infer_obs)
        # cv2.imshow("Predicted Next Obs",next_infer_obs)
        # cv2.waitKey(0)
        imgs.append(next_infer_obs)

    axes = axes.ravel()

    for ax in range(len(axes)):
        axes[ax].imshow(imgs[ax])                # 在子图上显示图片
        axes[ax].set_title(captions[ax])         # 为子图设置标题（注释）
        axes[ax].axis('off')                # 关闭坐标轴

    




    actions = ["Turn Left","Turn Right","Move Forward","OpenDoor"]
    c_action = hd_data["episodes"][evaluation_index]["actions"][c_time]

    if c_action == 5:
        c_action = 3

    print("Action " + actions[c_action] + " is taken in this timestep.") 

    plt.tight_layout()
    plt.show()




    # updating rnn hidden
    c_action = one_hot_encoding(c_action)
    c_action = torch.as_tensor(c_action, device=device).unsqueeze(0)

    # update rnn hidden and predict next state prior h_t+1 = f(h_t, s_t, a_t) , p(s_t+1 | h_t+1)
    _, rnn_hidden = rssm.prior(state,c_action,rnn_hidden) # s_t+1 ~ p(s_t+1 | h_t+1) , h_t+1 = f(h_t, s_t, a_t) update

    c_time += 1
    end_flg = int(input("next timestep?"))
    
   



    
