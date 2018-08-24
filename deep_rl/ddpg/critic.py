"""
Critic interface
"""

import copy
import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from ...common import FloatTensor, enable_cuda


class CriticNetwork(object):
    def __init__(self, critic, optimizer, tau):
        self.optimizer = optimizer
        self.tau = tau

        self.critic_network = critic
        self.target_critic_network = copy.deepcopy(critic)
        self.loss = MSELoss()

        if enable_cuda:
            self.critic_network.cuda()
            self.target_critic_network.cuda()
            self.loss.cuda()

    def train(self, inputs, action, predicted_q_value):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        predicted_q_value = torch.from_numpy(predicted_q_value).type(FloatTensor)

        self.optimizer.zero_grad()
        q_value = self.critic_network(inputs, action)
        output = self.loss(q_value, predicted_q_value)

        delta = (predicted_q_value - q_value).data.cpu().numpy()

        output.backward()
        self.optimizer.step()
        return q_value.data.cpu().numpy(), delta

    def predict(self, inputs, action):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        return self.critic_network(inputs, action).data.cpu().numpy()

    def predict_target(self, inputs, action):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        return self.target_critic_network(inputs, action).data.cpu().numpy()

    def action_gradients(self, inputs, actions):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        actions = Variable(torch.from_numpy(actions).type(FloatTensor), requires_grad=True)
        q_value = self.critic_network(inputs, actions)
        q_value = torch.mean(q_value)
        return torch.autograd.grad(q_value, actions)[0].data.cpu().numpy(), None

    def update_target_network(self):
        source = self.critic_network
        target = self.target_critic_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def gather_state_dict(self):
        state_dict = {
            'network': self.critic_network.state_dict(),
            'target': self.target_critic_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def scatter_state_dict(self, state_dict, all=True):
        self.critic_network.load_state_dict(state_dict['network'])
        self.target_critic_network.load_state_dict(state_dict['target'])
        if all:
            self.optimizer.load_state_dict(state_dict['optimizer'])
