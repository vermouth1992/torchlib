"""
Pytorch implementation of proximal policy optimization
"""

import argparse

from torchlib.common import enable_cuda

from .agent import Agent


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-3)
    parser.add_argument('--lam', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--target_kl', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--min_steps_per_batch', '-b', type=int, default=1000)

    parser.add_argument('--nn_size', '-s', type=int, default=64)


def get_nets(env, args):
    import gym
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    from torchlib.deep_rl.models.policy import CategoricalNNPolicy, BetaNNPolicy
    from torchlib.deep_rl.envs import is_atari_env, is_ple_game

    if len(env.observation_space.shape) == 1:
        # low dimensional environment
        if discrete:
            policy_net = CategoricalNNPolicy(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)
        else:
            policy_net = BetaNNPolicy(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

        if enable_cuda:
            policy_net.cuda()

        return policy_net

    elif is_atari_env(env.spec.id) or is_ple_game(env.spec.id):
        if env.observation_space.shape[:2] == (84, 84):
            frame_history_len = env.observation_space.shape[-1]

            from torchlib.deep_rl.models.policy import AtariPolicy
            policy_net = AtariPolicy(num_channel=frame_history_len, action_dim=env.action_space.n)

            if enable_cuda:
                policy_net.cuda()

            return policy_net
        else:
            raise ValueError('Not a typical env. Please define custom network')
    else:
        raise ValueError('Not a typical env. Please define custom network')
