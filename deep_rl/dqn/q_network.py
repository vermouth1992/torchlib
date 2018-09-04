"""
Q network for vanilla/double/dueling DQN.
Q learning only works for discrete action space Discrete() in gym.
"""

import copy

import numpy as np
import torch
from torch.nn import SmoothL1Loss
from torchlib.common import FloatTensor, LongTensor, enable_cuda


class QNetwork(object):
    def __init__(self, network, optimizer, tau):
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.optimizer = optimizer
        self.tau = tau
        self.loss = SmoothL1Loss()

        if enable_cuda:
            self.network.cuda()
            self.target_network.cuda()
            self.loss.cuda()

    def train(self, inputs, actions, predicted_q_value):
        """ train the q network with one step

        Args:
            inputs: state of shape (batch_size, state_dim), or (batch, channel, img_h, img_w)
            actions: shape of (batch_size,)
            predicted_q_value: (batch_size, )

        Returns: q_value from network

        """
        actions = torch.from_numpy(actions).type(LongTensor)
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        predicted_q_value = torch.from_numpy(predicted_q_value).type(FloatTensor)
        self.optimizer.zero_grad()
        q_value = self.network(inputs).gather(1, actions.unsqueeze(1)).squeeze()
        output = self.loss(q_value, predicted_q_value)
        output.backward()
        self.optimizer.step()
        delta = (predicted_q_value - q_value).data.cpu().numpy()
        return q_value.data.cpu().numpy(), delta

    def predict_action(self, inputs):
        return np.argmax(self.compute_q_value(inputs), axis=1)

    def compute_q_value(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        q_value = self.network(inputs)
        return q_value.data.cpu().numpy()

    def compute_target_q_value(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        q_value = self.target_network(inputs)
        return q_value.data.cpu().numpy()

    def update_target_network(self):
        source = self.network
        target = self.target_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def gather_state_dict(self):
        state_dict = {
            'network': self.network.state_dict(),
            'target': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def scatter_state_dict(self, state_dict, all=True):
        self.network.load_state_dict(state_dict['network'])
        self.target_network.load_state_dict(state_dict['target'])
        if all:
            self.optimizer.load_state_dict(state_dict['optimizer'])
