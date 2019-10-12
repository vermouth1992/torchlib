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
import time

import tensorlib.rl.algo as rl_algo
from tensorlib import rl
from tensorlib.utils.random import set_global_seeds

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running rl algorithms')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--frame_length', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=1992)
    # parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)

    algorithm_parsers = parser.add_subparsers(title='algorithm', help='algorithm specific parser', dest='algo')
    ppo_parser = algorithm_parsers.add_parser('ppo')
    rl_algo.ppo.add_args(ppo_parser)
    sac_parser = algorithm_parsers.add_parser('sac')
    rl_algo.sac.add_args(sac_parser)
    td3_parser = algorithm_parsers.add_parser('td3')
    rl_algo.td3.add_args(td3_parser)

    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    dummy_env = rl.envs.make_env(env_name=kwargs['env_name'],
                                 num_envs=None,
                                 frame_length=kwargs['frame_length'])

    # Set random seeds
    set_global_seeds(kwargs['seed'])

    # set logger directory
    logdir = '/tmp/experiments/{}_{}_{}_{}'.format(kwargs['env_name'], kwargs['exp_name'],
                                                   time.strftime("%m-%d-%Y_%H-%M-%S"), kwargs['seed'])

    if kwargs['algo'] == 'ppo':
        policy_net = rl_algo.ppo.get_nets(dummy_env, kwargs)
        agent = rl_algo.ppo.Agent(policy_net=policy_net, **kwargs)
    elif kwargs['algo'] == 'sac':
        nets = rl_algo.sac.get_nets(dummy_env, kwargs)
        agent = rl_algo.sac.Agent(nets=nets, action_space=dummy_env.action_space, **kwargs)
    elif kwargs['algo'] == 'td3':
        nets = rl_algo.td3.get_nets(dummy_env, kwargs)
        assert not rl.envs.is_discrete(dummy_env), 'TD3 only supports continuous environment'
        agent = rl_algo.td3.Agent(nets=nets, **kwargs)
    else:
        raise NotImplementedError

    if not kwargs['test']:
        del dummy_env
        env = rl.envs.make_env(env_name=kwargs['env_name'],
                               num_envs=kwargs['num_envs'],
                               frame_length=kwargs['frame_length'])
        env.seed(kwargs['seed'])
        agent.train(env=env, logdir=logdir, **kwargs)
    else:
        agent.load_checkpoint(kwargs['checkpoint_path'])
        rl.test(dummy_env, agent, num_episode=100, render=kwargs['render'], seed=kwargs['seed'])
