"""
Training DQN on Atari games
"""

import pprint

import gym
import torch

import torchlib.deep_rl as deep_rl
import torchlib.deep_rl.algorithm.value_based.dqn as dqn


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
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
    env_name = args['env_name']
    env = gym.make(env_name)

    env = deep_rl.envs.wrappers.wrap_deepmind(env)

    frame_history_len = 4

    if args['duel']:
        network = deep_rl.models.AtariDuelQModule(frame_history_len, action_dim=env.action_space.n)
    else:
        network = deep_rl.models.AtariQModule(frame_history_len, action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    num_iterations = float(args['n_iter']) / 4.0
    exploration_schedule = deep_rl.utils.schedules.PiecewiseSchedule(
        [
            (0, 1.0),
            (1e5, 0.02),
        ], outside_value=0.02
    )

    q_network = dqn.DQN(network, optimizer, optimizer_scheduler=None, tau=1e-3)
    checkpoint_path = 'checkpoint/{}.ckpt'.format(env_name)

    if args['test']:
        q_network.load_checkpoint(checkpoint_path)
        deep_rl.test(env, q_network, frame_history_len=frame_history_len, render=args['render'], seed=args['seed'])

    else:

        replay_buffer_config = {
            'size': args['replay_size'],
            'frame_history_len': frame_history_len
        }

        q_network.train(env, exploration_schedule, args['n_iter'], 'frame', replay_buffer_config,
                        batch_size=args['batch_size'], gamma=args['discount'], learn_starts=args['learn_start'],
                        learning_freq=args['learning_freq'], double_q=args['double_q'], seed=args['seed'],
                        log_every_n_steps=args['log_every_n_steps'], target_update_freq=args['target_update_freq'],
                        checkpoint_path=checkpoint_path)
