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
from models.utils import ReplayBuffer,one_hot_encoding,get_params

from torch.utils.tensorboard import SummaryWriter


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


def posterior(self, rnn_hidden, embedded_obs):
    """
    Compute posterior q(s_t | h_t, o_t)
    """
    hidden = self.act(self.fc_rnn_hidden_embedded_obs(
    torch.cat([rnn_hidden, embedded_obs], dim=1)))
    mean = self.fc_state_mean_posterior(hidden)
    stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
    return Normal(mean, stddev)


# parameters
run_name = "donefixed_compact"
chunk_length = 15
batch_size = 10


para = get_params()


state_dim = para["f_d"]
rnn_hidden_dim = para["h_d"]

lr = 0.0001

# define models and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().to(device)
rssm = RecurrentStateSpaceModel(state_dim,
                                4,
                                rnn_hidden_dim).to(device)
obs_model = ObservationModel(state_dim, rnn_hidden_dim).to(device)
all_params = (list(encoder.parameters()) +
              list(rssm.parameters()) +
              list(obs_model.parameters()))
optimizer = Adam(all_params, lr)

writer = SummaryWriter('mb_runs/mb_tril_' + run_name)


# wrap the HD data into RB

replay_buffer = ReplayBuffer(capacity=1000,
                            observation_shape=(64,64,3),
                            action_dim=4) # replay buffer

with open('human_demo_data/hd_minigrid.json', 'r') as file:
    hd_data = json.load(file)


for eps in range(len(hd_data["episodes"])):
    num_actions = len(hd_data["episodes"][eps]["actions"])
    for timestep in range(num_actions + 1):

        # get observation
        with h5py.File('human_demo_data/' + str(eps) + "_imgSeq/imgSeq", 'r') as hdf5_file:
            image_data = hdf5_file["imgSeq"][timestep]
            image_data = cv2.resize(image_data,(64,64),interpolation=cv2.INTER_LINEAR)
            if timestep == num_actions: # last action
                replay_buffer.push(image_data, one_hot_encoding(-1), 1)
            else:
                replay_buffer.push(image_data, one_hot_encoding(hd_data["episodes"][eps]["actions"][timestep]), 0)



# Training World Model_____________________________________________________________

for update_step in range(20000):

    observations, actions, _ =  replay_buffer.sample(batch_size, chunk_length)

    # preprocess observations and transpose tensor for RNN training
    observations = preprocess_obs(observations)
    observations = torch.as_tensor(observations, device=device)
    observations = observations.transpose(3, 4).transpose(2, 3)
    observations = observations.transpose(0, 1)
    actions = torch.as_tensor(actions, device=device).transpose(0, 1)




    # embed observations with CNN
    embedded_observations = encoder(
    observations.reshape(-1, 3, 64, 64)).view(chunk_length, batch_size, -1)

    # prepare Tensor to maintain states sequence and rnn hidden states sequence
    states = torch.zeros(
    chunk_length, batch_size, state_dim, device=device)
    rnn_hiddens = torch.zeros(
    chunk_length, batch_size, rnn_hidden_dim, device=device)


    # initialize state and rnn hidden state with 0 vector
    state = torch.zeros(batch_size, state_dim, device=device)
    rnn_hidden = torch.zeros(batch_size, rnn_hidden_dim, device=device)




    # compute state and rnn hidden sequences and kl loss (averaged across chunks)
    kl_loss = 0
    for l in range(chunk_length - 1):
        next_state_prior, next_state_posterior, rnn_hidden = \
            rssm(state, actions[l], rnn_hidden, embedded_observations[l + 1])
        state = next_state_posterior.rsample()
        states[l + 1] = state
        rnn_hiddens[l + 1] = rnn_hidden
        kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
        kl_loss += kl.clamp(min=1).mean()
    kl_loss /= (chunk_length - 1)




    # compute reconstructed observations and predicted rewards (forward computation)
    flatten_states = states.view(-1, state_dim)
    flatten_rnn_hiddens = rnn_hiddens.view(-1, rnn_hidden_dim)
    recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(chunk_length, batch_size, 3, 64, 64)




    # compute loss for observation and reward
    obs_loss = mse_loss(
        recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()


    # add all losses and update model parameters with gradient descent
    loss = kl_loss + obs_loss
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(all_params, 1000)
    optimizer.step()


    # print losses and add tensorboard
    print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f'
        % (update_step + 1,
        loss.item(), kl_loss.item(), obs_loss.item()))
    
    writer.add_scalar('overall loss', loss.item(), update_step)
    writer.add_scalar('kl loss', kl_loss.item(), update_step)
    writer.add_scalar('obs loss', obs_loss.item(), update_step)


    if (update_step + 1) % 1000 == 0 and update_step != 0:
        torch.save(encoder.state_dict(), "encoder_" + str(update_step + 1) + ".pth")
        torch.save(rssm.state_dict(), "rssm_" + str(update_step + 1) + ".pth")
        torch.save(obs_model.state_dict(),"obs_model_" + str(update_step + 1) + ".pth")



    

