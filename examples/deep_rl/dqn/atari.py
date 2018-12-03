"""
Training DQN on Atari games
"""

import os
import pprint

import gym
import torch
import torch.nn as nn
import torchlib.deep_rl.value_based.dqn as dqn
from gym import wrappers
from torchlib.deep_rl.utils.atari_wrappers import wrap_deepmind
from torchlib.deep_rl.utils.schedules import PiecewiseSchedule
from torchlib.deep_rl.value_based.dqn import QNetwork
from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block, linear_bn_relu_block, Flatten


class QModule(nn.Module):
    def __init__(self, action_dim):
        super(QModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(3, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=3, stride=2, padding=1, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(10 * 10 * 64, 512, normalize=False),
            *linear_bn_relu_block(512, action_dim, normalize=False)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.model.forward(x)
        return x


class DuelQModule(nn.Module):
    def __init__(self, action_dim):
        super(DuelQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(3, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=3, stride=2, padding=1, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(10 * 10 * 64, 512, normalize=False),
            *linear_bn_relu_block(512, action_dim, normalize=False)
        )
        self.adv_fc = nn.Linear(512, action_dim)
        self.value_fc = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = self.model.forward(x)
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

    env = wrap_deepmind(env)

    if args['duel']:
        network = DuelQModule(action_dim=env.action_space.n)
    else:
        network = QModule(action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    num_iterations = float(args['n_iter']) / 4.0
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    lr_multiplier = 1.0

    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    lr_schedule_lambda = lambda t: lr_schedule.value(t)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_lambda)

    q_network = QNetwork(network, optimizer, scheduler, tau=1e-3)
    checkpoint_path = 'checkpoint/{}.ckpt'.format(env_name)

    if args['test']:
        try:
            q_network.load_checkpoint(checkpoint_path)
            dqn.test(env, q_network, seed=args['seed'])
        except:
            print("Can't find checkpoint. Abort")

    else:

        replay_buffer_config = {
            'size': args['replay_size'],
            'frame_history_len': 4
        }

        dqn.train(env, q_network, exploration_schedule, args['n_iter'], 'frame', replay_buffer_config,
                  batch_size=args['batch_size'], gamma=args['discount'], learn_starts=args['learn_start'],
                  learning_freq=args['learning_freq'], double_q=args['double_q'], seed=args['seed'],
                  log_every_n_steps=args['log_every_n_steps'], checkpoint_path=checkpoint_path)
