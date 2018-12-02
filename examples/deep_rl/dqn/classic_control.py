"""
Train agent on CartPole-v0 using DQN
"""

import os
import pprint

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchlib.deep_rl.value_based.dqn as dqn
from gym import wrappers
from torchlib.deep_rl.utils.schedules import PiecewiseSchedule
from torchlib.deep_rl.value_based.dqn import QNetwork


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


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100000)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--learning_freq', '-lf', type=int, default=1)
    parser.add_argument('--replay_type', type=str, default='normal')
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--learn_start', type=int, default=1000)
    parser.add_argument('--duel', action='store_true')
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    env_name = args['env_name']
    env = gym.make(env_name)

    expt_dir = '/tmp/{}'.format(env_name)
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

    if args['duel']:
        network = DuelQModule(args['nn_size'], state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    else:
        network = QModule(args['nn_size'], state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    q_network = QNetwork(network, optimizer)
    checkpoint_path = 'checkpoint/{}.ckpt'.format(env_name)

    if args['test']:
        try:
            q_network.load_checkpoint(checkpoint_path)
            dqn.test(env, q_network, seed=args['seed'])
        except:
            print("Can't find checkpoint. Abort")

    else:
        exploration_criteria = PiecewiseSchedule(
            [
                (0, 1.0),
                (args['n_iter'] // 10, 0.02),
            ], outside_value=0.02
        )

        replay_buffer_config = {
            'size': args['replay_size'],
        }

        dqn.train(env, q_network, exploration_criteria, args['n_iter'], args['replay_type'], replay_buffer_config,
                  batch_size=args['batch_size'], gamma=args['discount'], learn_starts=args['learn_start'],
                  double_q=args['double_q'], seed=args['seed'], log_every_n_steps=args['log_every_n_steps'],
                  checkpoint_path=checkpoint_path)
