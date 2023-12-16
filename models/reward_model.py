import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    Reward model to predict reward from state and rnn hidden state
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=300, act=F.relu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward
