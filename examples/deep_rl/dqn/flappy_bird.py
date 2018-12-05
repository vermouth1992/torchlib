"""
Train DQN to play flappy bird
"""

import os
import pprint

import cv2
import gym
import gym_ple
import numpy as np
import torch
import torch.nn as nn
import torchlib.deep_rl.value_based.dqn as dqn
from gym import spaces
from gym import wrappers
from torchlib.deep_rl.utils.atari_wrappers import MaxAndSkipEnv, ClippedRewardsWrapper
from torchlib.deep_rl.utils.schedules import PiecewiseSchedule
from torchlib.deep_rl.value_based.dqn import QNetwork
from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block, linear_bn_relu_block, Flatten


def _process_frame_flappy_bird(frame):
    img = np.reshape(frame, [512, 288, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 149), interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[0:84, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)


class FlappyBirdNoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=10):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(FlappyBirdNoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs = self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(self.env.action_space.sample())
        return obs


class ProcessFrameFlappyBird(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameFlappyBird, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame_flappy_bird(obs), reward, done, info

    def _reset(self):
        return _process_frame_flappy_bird(self.env.reset())


def wrap_flappybird(env):
    env = FlappyBirdNoopResetEnv(env, noop_max=10)
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrameFlappyBird(env)
    env = ClippedRewardsWrapper(env)
    return env


class QModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(QModule, self).__init__()
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


class DuelQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(DuelQModule, self).__init__()
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


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100000)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--learning_freq', '-lf', type=int, default=2)
    parser.add_argument('--replay_size', type=int, default=500000)
    parser.add_argument('--learn_start', type=int, default=5000)
    parser.add_argument('--duel', action='store_true')
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--target_update_freq', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', action='store_true')
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

    if args['duel']:
        network = DuelQModule(frame_history_len, action_dim=env.action_space.n)
    else:
        network = QModule(frame_history_len, action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    num_iterations = float(args['n_iter']) / 4.0
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e5, 0.1),
            (1e6, 0.02)
        ], outside_value=0.02
    )

    # lr_multiplier = 1.0
    #
    # lr_schedule = PiecewiseSchedule([
    #     (0, 1e-4 * lr_multiplier),
    #     (num_iterations / 10, 1e-4 * lr_multiplier),
    #     (num_iterations / 2, 5e-5 * lr_multiplier),
    # ],
    #     outside_value=5e-5 * lr_multiplier)
    #
    # lr_schedule_lambda = lambda t: lr_schedule.value(t)
    #
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_lambda)

    q_network = QNetwork(network, optimizer, optimizer_scheduler=None, tau=1e-3)
    checkpoint_path = 'checkpoint/{}.ckpt'.format(env_name)

    if args['test']:
        q_network.load_checkpoint(checkpoint_path)
        dqn.test(env, q_network, frame_history_len=frame_history_len, render=args['render'], seed=args['seed'])

    else:

        replay_buffer_config = {
            'size': args['replay_size'],
            'frame_history_len': frame_history_len
        }

        dqn.train(env, q_network, exploration_schedule, args['n_iter'], 'frame', replay_buffer_config,
                  batch_size=args['batch_size'], gamma=args['discount'], learn_starts=args['learn_start'],
                  learning_freq=args['learning_freq'], double_q=args['double_q'], seed=args['seed'],
                  log_every_n_steps=args['log_every_n_steps'], target_update_freq=args['target_update_freq'],
                  checkpoint_path=checkpoint_path)
