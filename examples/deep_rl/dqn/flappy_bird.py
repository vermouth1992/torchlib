"""
Train DQN to play flappy bird
"""

import os
import pprint

import gym_ple
import torch.optim
import torchlib.deep_rl.value_based.dqn as dqn
from gym import wrappers
from torchlib import deep_rl
from torchlib.deep_rl.envs.wrappers import wrap_flappybird
from torchlib.deep_rl.models.value import AtariQModule, AtariDuelQModule
from torchlib.deep_rl.utils.schedules import PiecewiseSchedule
from torchlib.deep_rl.value_based.dqn import QNetwork


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
        network = AtariDuelQModule(frame_history_len, action_dim=env.action_space.n)
    else:
        network = AtariQModule(frame_history_len, action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    num_iterations = float(args['n_iter']) / 4.0
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (5e4, 0.01),
            (1e5, 0.005),
            (1e6, 0.001)
        ], outside_value=0.001
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
        deep_rl.test(env, q_network, frame_history_len=1, render=args['render'], seed=args['seed'])

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
