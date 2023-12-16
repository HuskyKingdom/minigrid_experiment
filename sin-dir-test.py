# models
from models.encoder import Encoder
from models.RSSM import RecurrentStateSpaceModel
from models.utils import show_seq,contrastive_dataloader,restore_obs,get_params
from models.observation_model import ObservationModel

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
    test_index = 21


    language_encoder = CTR_LanguageEncoder().to(device)
    jmf_encoder = CTR_JmfEncoder(state_dim,hidden_dim,lstm_layers).to(device)

    run_name = para["encoders"] + ep

    language_encoder.load_state_dict(torch.load("language_encoder_" + run_name + ".pth"))
    jmf_encoder.load_state_dict(torch.load("jmf_encoder_" + run_name + ".pth"))

    language_encoder.eval()
    jmf_encoder.eval()

    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(state_dim,
                            4,
                            hidden_dim).to(device)
    obs_model = ObservationModel(state_dim,hidden_dim).to(device)

    encoder.load_state_dict(torch.load("encoder_" + para["rssms"] + ".pth"))
    rssm.load_state_dict(torch.load("rssm_" + para["rssms"] + ".pth"))
    obs_model.load_state_dict(torch.load("obs_model_" + para["rssms"] + ".pth"))

    encoder.eval()
    rssm.eval()
    obs_model.eval()

    rnn_hidden = torch.zeros(1, hidden_dim, device=device)
    img_seq = None
    encoded_seq = []

    with open('human_demo_data/hd_minigrid.json', 'r') as file:
        hd_data = json.load(file)


    for i in range(len(hd_data["episodes"][test_index]["actions"])+1):

        with h5py.File('human_demo_data/' + str(test_index) + "_imgSeq/imgSeq", 'r') as hdf5_file:
            image_data = hdf5_file["imgSeq"][i] # current obs
            image_data = cv2.resize(image_data,(64,64),interpolation=cv2.INTER_LINEAR)

        obs = image_data
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        

        with torch.no_grad():
            embedded_obs = encoder(obs)
            state_posterior = rssm.posterior(rnn_hidden, embedded_obs)
            state = state_posterior.mean # s_t ~ p(s_t | h_t, o_t)
            


            if img_seq == None:
                img_seq = state
            else:
                img_seq = torch.cat((img_seq,state),dim=0)

            infer_obs = obs_model(state,rnn_hidden).squeeze(0).transpose(0, 1).transpose(1, 2)
            infer_obs = infer_obs.detach().cpu().numpy()
            encoded_seq.append(restore_obs(infer_obs)) 
            
            
            if i == len(hd_data["episodes"][test_index]["actions"]): # last obs
                current_action = 3
            else:
                current_action = hd_data["episodes"][test_index]["actions"][i]
                if current_action == 5:
                    current_action = 3
            
            action_emb = torch.as_tensor(one_hot[current_action], device=device).unsqueeze(0)
            _, rnn_hidden = rssm.prior(state,action_emb,rnn_hidden) # update rnn hidden


    img_seq = img_seq.unsqueeze(0)

    real_target = hd_data["episodes"][test_index]["language"]

    languages = [real_target,"go to a door and go to a green box", "go to a blue door and go to a gray door", "go to the red box","go to the red box and go to grey door"]


    language_emb = language_encoder(languages)
    state_emb = jmf_encoder(img_seq,None)


    language_emb = F.normalize(language_emb, p=2, dim=1)
    state_emb = F.normalize(state_emb, p=2, dim=1)


    # compute loss
    similarity_matrix_lan = torch.matmul(language_emb, state_emb.T) / 0.1 # cos 



    print(similarity_matrix_lan)
    print(languages)

    show_seq(encoded_seq,hd_data["episodes"][test_index]["actions"])

    

