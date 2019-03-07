"""
Common architecture for policy gradient
"""

import torch.nn as nn
import torch

from torchlib.common import FloatTensor

class PolicyDiscrete(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim, recurrent=False, hidden_size=20):
        assert action_dim > 1, 'Action dim must be greater than 1. Got {}'.format(action_dim)
        super(PolicyDiscrete, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU()
        )

        if recurrent:
            linear_size = hidden_size
        else:
            linear_size = nn_size

        self.action_head = nn.Sequential(
            nn.Linear(linear_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(linear_size, 1)

        self.recurrent = recurrent

        if self.recurrent:
            self.gru = nn.GRU(nn_size, hidden_size)

    def forward(self, state, hidden):
        x = self.model.forward(state)
        if self.recurrent:
            x, hidden = self.gru.forward(x.unsqueeze(0), hidden.unsqueeze(0))
            x = x.squeeze(0)
            hidden = hidden.squeeze(0)
        action = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return action, hidden, value.squeeze(-1)


class PolicyContinuous(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim, recurrent=False, hidden_size=20):
        super(PolicyContinuous, self).__init__()
        self.logstd = torch.nn.Parameter(torch.randn([action_dim, action_dim], requires_grad=True).type(FloatTensor))
        self.model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU(),
        )

        if recurrent:
            linear_size = hidden_size
        else:
            linear_size = nn_size

        self.action_head = nn.Sequential(
            nn.Linear(linear_size, action_dim),
            nn.Tanh()
        )
        self.value_head = nn.Linear(linear_size, 1)

        self.recurrent = recurrent

        if self.recurrent:
            self.gru = nn.GRU(nn_size, hidden_size)

    def forward(self, state, hidden):
        x = self.model.forward(state)
        if self.recurrent:
            x, hidden = self.gru.forward(x.unsqueeze(0), hidden.unsqueeze(0))
            x = x.squeeze(0)
            hidden = hidden.squeeze(0)
        mean = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return (mean, self.logstd), hidden, value.squeeze(-1)