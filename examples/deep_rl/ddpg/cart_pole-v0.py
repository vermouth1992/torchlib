"""
Use DDPG to solve CartPole-v0
"""
from __future__ import print_function, division

import argparse
import pprint
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.deep_rl.ddpg import ActorNetwork, CriticNetwork, Trainer
from torchlib.utils.random.random_process import OrnsteinUhlenbeckActionNoise
from torchlib.utils.weight_utils import fanin_init


class ActorModule(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(400, 300)
        fanin_init(self.fc2)
        self.fc3 = nn.Linear(300, action_dim)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        self.action_bound = action_bound

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
    def __init__(self, state_dim, action_dim):
        super(CriticModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(256, 256)
        fanin_init(self.fc2)
        self.fc_action = nn.Linear(action_dim, 256)
        fanin_init(self.fc_action)
        self.fc3 = nn.Linear(256, 1)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent for CartPole-v0')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--episode', required='--train' in sys.argv)

    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']

    env = gym.make('CartPole-v0')

    config = {
        'seed': 1996,
        'max step': 1000,
        'batch size': 64,
        'buffer size': 100000,
        'gamma': 0.99,
        'use_prioritized_buffer': True
    }


    def action_processor(action):
        if action > 0:
            return 1
        else:
            return 0


    actor = ActorModule(state_dim=4, action_dim=1, action_bound=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic = CriticModule(state_dim=4, action_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    actor = ActorNetwork(actor, optimizer=actor_optimizer, tau=1e-3)
    critic = CriticNetwork(critic, optimizer=critic_optimizer, tau=1e-3)
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1))
    ddpg_model = Trainer(env, actor, critic, actor_noise, config, action_processor=action_processor)
    checkpoint_path = './checkpoint/cart_pole-v0.ckpt'
    num_episode = int(args['episode'])

    if train:
        resume = args['resume']
        if resume == 'model':
            ddpg_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            ddpg_model.load_checkpoint(checkpoint_path, all=True)

        ddpg_model.train(num_episode, checkpoint_path, save_every_episode=-1)
    else:
        ddpg_model.load_checkpoint(checkpoint_path, all=False)
        ddpg_model.test(num_episode)
