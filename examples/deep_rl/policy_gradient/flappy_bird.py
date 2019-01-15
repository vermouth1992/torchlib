"""
Solve Flappy bird using PPO
"""

"""
Train DQN to play flappy bird
"""

import os
import pprint

import gym_ple
import torch
import torch.nn as nn
import torchlib.deep_rl.policy_gradient.ppo as ppo
from deep_rl.envs.wrappers import wrap_flappybird
from gym import wrappers
from torchlib import deep_rl
from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block, linear_bn_relu_block, Flatten


class Policy(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(Policy, self).__init__()
        self.feature = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
        )
        self.action_head = nn.Sequential(
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.feature.forward(x)
        action = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return action, value.squeeze(-1)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=0.0)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--batch_size', '-b', type=int, default=10000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    env_name = 'FlappyBird-v0'
    env = gym_ple.make(env_name)

    expt_dir = '/tmp/{}'.format(env_name)
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

    env = wrap_flappybird(env)

    frame_history_len = 4

    network = Policy(frame_history_len, action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    gae_lambda = args['gae_lambda']
    max_path_length = args['ep_len'] if args['ep_len'] > 0 else None

    agent = ppo.Agent(network, optimizer, True, gae_lambda, clip_param=args['clip_param'],
                      entropy_coef=args['entropy_coef'], value_coef=args['value_coef'])

    checkpoint_path = 'checkpoint/{}.ckpt'.format(env_name)

    if args['test']:
        agent.load_checkpoint(checkpoint_path)
        deep_rl.test(env, agent, frame_history_len=frame_history_len, render=args['render'], seed=args['seed'])

    else:
        ppo.train(args['exp_name'], env, agent, args['n_iter'], args['discount'], args['batch_size'], max_path_length,
                  logdir=None, seed=args['seed'], checkpoint_path=checkpoint_path)
