"""
Use DDPG to solve low dimensional control
"""
from __future__ import print_function, division

import os
import pprint

import gym.spaces
import numpy as np
import torch
import torchlib.deep_rl.value_based.ddpg as ddpg
from gym import wrappers
from torchlib import deep_rl
from torchlib.deep_rl.value_based.ddpg import ActorNetwork, CriticNetwork
from torchlib.utils.random.random_process import OrnsteinUhlenbeckActionNoise



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
    parser.add_argument('--target_update_tau', type=float, default=1e-3)
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()

    args = vars(parser.parse_args())
    pprint.pprint(args)

    env_name = args['env_name']
    env = gym.make(env_name)

    expt_dir = '/tmp/{}'.format(env_name)
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    action_bound = env.action_space.high
    print('Action space high: {}'.format(env.action_space.high))
    print('Action space low: {}'.format(env.action_space.low))
    assert np.all(env.action_space.high == -env.action_space.low), 'Check the action space.'

    nn_size = args['nn_size']
    tau = args['target_update_tau']

    actor = ActorModule(size=nn_size, state_dim=ob_dim, action_dim=ac_dim, action_bound=action_bound)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args['learning_rate'])
    critic = CriticModule(size=nn_size, state_dim=ob_dim, action_dim=ac_dim)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args['learning_rate'])

    actor = ActorNetwork(actor, optimizer=actor_optimizer, tau=tau)
    critic = CriticNetwork(critic, optimizer=critic_optimizer, tau=tau)
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ac_dim))

    checkpoint_path = 'checkpoint/{}.ckpt'.format(env_name)

    if args['test']:
        try:
            actor.load_checkpoint(checkpoint_path)
            deep_rl.test(env, actor, seed=args['seed'])
        except:
            print("Can't find checkpoint. Abort")

    else:

        replay_buffer_config = {
            'size': args['replay_size'],
        }

        ddpg.train(env, actor, critic, actor_noise, args['n_iter'], args['replay_type'],
                   replay_buffer_config, args['batch_size'], args['discount'], args['learn_start'],
                   learning_freq=args['learning_freq'], seed=args['seed'], log_every_n_steps=args['log_every_n_steps'],
                   actor_checkpoint_path=checkpoint_path)
