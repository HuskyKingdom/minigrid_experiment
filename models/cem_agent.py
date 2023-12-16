import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam


from models.encoder import Encoder
from models.reward_model import RewardModel
from models.observation_model import ObservationModel
from models.RSSM import RecurrentStateSpaceModel



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


class CEMAgent:
    """
    Action planning by Cross Entropy Method (CEM) in learned RSSM Model
    """
    def __init__(self, encoder, rssm, reward_model,
                 horizon, N_iterations, N_candidates, N_top_candidates):
        self.encoder = encoder
        self.rssm = rssm
        self.reward_model = reward_model

        self.horizon = horizon # 12
        self.N_iterations = N_iterations # 10
        self.N_candidates = N_candidates # 1000
        self.N_top_candidates = N_top_candidates # 100

        self.device = next(self.reward_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs):

        # Preprocess observation and transpose for torch style (channel-first)
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            
            # Compute starting state for planning
            # while taking information from current observation (posterior)
            embedded_obs = self.encoder(obs)

            # 先根据RNN隐藏层信息和编码后的观测信息计算状态的后验分布。q(s_t | h_t, o_t)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)

            # Initialize action distribution
            # 对应于论文的伪代码"Initialize factorized belief over action sequences q(a t:t+H 			 # ) ← Normal(0, I)."
            action_dist = Normal(
                torch.zeros((self.horizon, self.rssm.action_dim), device=self.device),
                torch.ones((self.horizon, self.rssm.action_dim), device=self.device)
            )

            # Iteratively improve action distribution with CEM
            for itr in range(self.N_iterations):
                # Sample action candidates and transpose to
                # (self.horizon, self.N_candidates, action_dim) for parallel exploration
                action_candidates = \
                    action_dist.sample([self.N_candidates]).transpose(0, 1)

                # Initialize reward, state, and rnn hidden state
                # The size of state is (self.N_acndidates, state_dim)
                # The size of rnn hidden is (self.N_candidates, rnn_hidden_dim)
                # These are for parallel exploration
                total_predicted_reward = torch.zeros(self.N_candidates, 				device=self.device)
                state = state_posterior.sample([self.N_candidates]).squeeze()
                rnn_hidden = self.rnn_hidden.repeat([self.N_candidates, 1])

                # Compute total predicted reward by open-loop prediction using prior p(s_t+1 | h_t+1)
                for t in range(self.horizon):
                    next_state_prior, rnn_hidden = \
                        self.rssm.prior(state, action_candidates[t], rnn_hidden)
                    # 此处 state 一直被赋值，因此是多步开环的预测
                    state = next_state_prior.sample()
                    total_predicted_reward += self.reward_model(state, rnn_hidden).squeeze()

                # update action distribution using top-k samples
                top_indexes = \
                    total_predicted_reward.argsort(descending=True)[: self.N_top_candidates]
                top_action_candidates = action_candidates[:, top_indexes, :]
                mean = top_action_candidates.mean(dim=1)
                stddev = (top_action_candidates - mean.unsqueeze(1)
                          ).abs().sum(dim=1) / (self.N_top_candidates - 1)
                action_dist = Normal(mean, stddev)

        # Return only first action (replan each state based on new observation)
        action = mean[0]

        # update rnn hidden state for next step planning
        with torch.no_grad():
            _, self.rnn_hidden = self.rssm.prior(state_posterior.sample(),
                                                 action.unsqueeze(0),
                                                 self.rnn_hidden)
        return action.cpu().numpy()

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
