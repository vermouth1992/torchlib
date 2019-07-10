"""
Solve Flappy bird using PPO
"""

"""
Train PPO to play flappy bird
"""

import numpy as np
import os
import pprint
import gym

import torch.optim
import torchlib.deep_rl.policy_gradient.ppo as ppo
from torchlib.deep_rl.envs.wrappers import wrap_flappybird, wrap_deepmind
from gym import wrappers
from torchlib import deep_rl

from torchlib.deep_rl.models.policy import AtariPolicy


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=0.001)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=4000)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--recurrent', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    env_name = args['env_name']

    if env_name == 'FlappyBird-v0':
        import gym_ple

        wrapper = wrap_flappybird
        env = gym_ple.make(env_name)

    else:
        wrapper = wrap_deepmind
        env = gym.make(env_name)

    expt_dir = '/tmp/{}'.format(env_name)
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

    recurrent = args['recurrent']
    hidden_size = args['hidden_size']

    if recurrent:
        frame_history_len = 1
        init_hidden_unit = np.zeros(shape=(hidden_size))
    else:
        frame_history_len = 4
        init_hidden_unit = None

    env = wrapper(env, frame_length=frame_history_len)

    network = AtariPolicy(recurrent=recurrent, hidden_size=hidden_size,
                          num_channel=frame_history_len, action_dim=env.action_space.n)

    optimizer = torch.optim.Adam(network.parameters(), lr=args['learning_rate'])

    gae_lambda = args['gae_lambda']
    max_path_length = args['ep_len'] if args['ep_len'] > 0 else None

    agent = ppo.PPOAgent(network, optimizer,
                         init_hidden_unit=init_hidden_unit,
                         lam=gae_lambda, clip_param=args['clip_param'],
                         entropy_coef=args['entropy_coef'], value_coef=args['value_coef'])

    checkpoint_path = 'checkpoint/{}_{}.ckpt'.format(env_name, recurrent)

    if args['test']:
        agent.load_checkpoint(checkpoint_path)
        deep_rl.test(env, agent, num_episode=args['n_iter'], frame_history_len=1, render=args['render'],
                     seed=args['seed'])

    else:
        ppo.train(args['exp_name'], env, agent, args['n_iter'], args['discount'], args['batch_size'], max_path_length,
                  logdir=None, seed=args['seed'], checkpoint_path=checkpoint_path)
