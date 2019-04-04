"""
Typical modules for value functions and q values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block, linear_bn_relu_block, Flatten
from torchlib.utils.weight_utils import fanin_init

"""
Low dimensional classic control module
"""


class QModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(QModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DuelQModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(DuelQModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        self.fc2 = nn.Linear(size, size)
        self.adv_fc = nn.Linear(size, action_dim)
        self.value_fc = nn.Linear(size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        x = value + adv
        return x


class ActorModule(nn.Module):
    def __init__(self, size, state_dim, action_dim, action_bound):
        super(ActorModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(size, size)
        fanin_init(self.fc2)
        self.fc3 = nn.Linear(size, action_dim)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        self.action_bound = torch.tensor(action_bound, requires_grad=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = x * self.action_bound
        return x


class CriticModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(CriticModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(size, size)
        fanin_init(self.fc2)
        self.fc_action = nn.Linear(action_dim, size)
        fanin_init(self.fc_action)
        self.fc3 = nn.Linear(size, 1)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = F.relu(x)
        x_state = self.fc2(x)
        x_action = self.fc_action(action)
        x = torch.add(x_state, x_action)
        x = F.relu(x)
        x = self.fc3(x)
        return x


"""
Atari Policy
"""


class AtariQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(AtariQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.model.forward(x)
        return x


class AtariDuelQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(AtariDuelQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
        )
        self.adv_fc = nn.Linear(512, action_dim)
        self.value_fc = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.model.forward(x)
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        x = value + adv
        return x
