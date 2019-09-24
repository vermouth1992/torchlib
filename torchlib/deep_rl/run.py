"""
Runner class for standard gym environments to test the performance of basic RL algorithms. These environments including
1. Low dimensional environment
2. Atari Games
For each algorithm, then need to have the following components
- A default argument parser
- A default method to create networks
- Instantiate agent
- Train the agent
"""

import argparse
import pprint

import torchlib.deep_rl.algorithm as rl_algo
from torchlib import deep_rl

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running rl algorithms')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--frame_length', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)

    algorithm_parsers = parser.add_subparsers(title='algorithm', help='algorithm specific parser', dest='algo')
    ppo_parser = algorithm_parsers.add_parser('ppo')
    rl_algo.ppo.add_args(ppo_parser)

    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    dummy_env = deep_rl.envs.make_env(env_name=kwargs['env_name'],
                                      num_envs=None,
                                      frame_length=kwargs['frame_length'])

    if kwargs['algo'] == 'ppo':
        policy_net = rl_algo.ppo.get_nets(dummy_env, kwargs)
        agent = rl_algo.ppo.Agent(policy_net=policy_net, **kwargs)
    else:
        raise NotImplementedError

    del dummy_env
    env = deep_rl.envs.make_env(env_name=kwargs['env_name'],
                                num_envs=kwargs['num_envs'],
                                frame_length=kwargs['frame_length'])

    if not kwargs['test']:
        agent.train(env=env, **kwargs)
