"""
Model for approximate system dynamics
"""

import torch
import torch.nn as nn

from torchlib.utils.layers import linear_bn_relu_block


class MLPDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, nn_size=64):
        super(MLPDynamics, self).__init__()
        self.model = nn.Sequential(
            *linear_bn_relu_block(state_dim + action_dim, nn_size, normalize=False),
            *linear_bn_relu_block(nn_size, nn_size, normalize=False),
            nn.Linear(nn_size, state_dim)
        )

    def forward(self, states, actions):
        state_action_input = torch.cat((states, actions), dim=-1)
        return self.model(state_action_input)
