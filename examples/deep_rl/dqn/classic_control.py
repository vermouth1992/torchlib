"""
Train agent on CartPole-v0 using DQN
"""

import argparse
import pprint
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.deep_rl.dqn import QNetwork, Trainer
from torchlib.deep_rl.utils.exploration import EpsilonGreedy


class QModule(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DQN agent for classic controls')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--episode', required='--train' in sys.argv)
    parser.add_argument('--env', required=True, choices=['CartPole-v0', 'Acrobot-v1'])

    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    env_name = args['env']
    print('Environment: {}'.format(env_name))

    env = gym.make(env_name)

    config = {
        'seed': 0,
        'max step': 1000,
        'batch size': 64,
        'buffer size': 10000,
        'gamma': 0.99,
        'double_q': True,
        'use_prioritized_buffer': False
    }

    pprint.pprint(config)

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    network = QModule(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    q_network = QNetwork(network, optimizer, tau=1e-3)

    checkpoint_path = './checkpoint/{}.ckpt'.format(env_name)
    num_episode = int(args['episode'])

    exploration_criteria = EpsilonGreedy(epsilon=1., decay=1e-4, minimum=0.01)

    trainer = Trainer(env, q_network, exploration_criteria, config)

    if train:
        resume = args['resume']
        if resume == 'model':
            trainer.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            trainer.load_checkpoint(checkpoint_path, all=True)

        trainer.train(num_episode, checkpoint_path, save_every_episode=-1)
    else:
        trainer.load_checkpoint(checkpoint_path, all=False)
        trainer.test(num_episode)
