"""
Actor interface for all ddpg models
"""

import copy
import torch

from lib.common import FloatTensor, enable_cuda


class ActorNetwork(object):
    def __init__(self, actor, optimizer, tau):
        self.tau = tau
        self.optimizer = optimizer

        self.actor_network = actor
        self.target_actor_network = copy.deepcopy(actor)
        if enable_cuda:
            self.actor_network.cuda()
            self.target_actor_network.cuda()

    def train(self, inputs, a_gradient):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        a_gradient = torch.from_numpy(a_gradient).type(FloatTensor)
        self.optimizer.zero_grad()
        actions = self.actor_network(inputs)
        actions.backward(-a_gradient)
        self.optimizer.step()

    def predict(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        return self.actor_network(inputs).data.cpu().numpy()

    def predict_target(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        return self.target_actor_network(inputs).data.cpu().numpy()

    def update_target_network(self):
        source = self.actor_network
        target = self.target_actor_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def gather_state_dict(self):
        state_dict = {
            'network': self.actor_network.state_dict(),
            'target': self.target_actor_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def scatter_state_dict(self, state_dict, all=True):
        self.actor_network.load_state_dict(state_dict['network'])
        self.target_actor_network.load_state_dict(state_dict['target'])
        if all:
            self.optimizer.load_state_dict(state_dict['optimizer'])
