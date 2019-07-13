"""
Test Vanilla PG on standard environment, where state is (ob_dim) and action is continuous/discrete

"""

import pprint

import numpy as np
import torch.optim

import torchlib.deep_rl.policy_gradient as pg
import torchlib.deep_rl.policy_gradient.a2c as a2c
from torchlib import deep_rl
from torchlib.deep_rl.envs import make_env, is_ple_game, is_atari_env
from torchlib.utils.random import set_global_seeds

# used for import self-defined envs
__all__ = ['deep_rl']

if __name__ == '__main__':
    parser = a2c.make_default_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    env_name = args['env_name']

    assert args['discount'] < 1.0, 'discount must be smaller than 1.0'
    set_global_seeds(args['seed'])

    print('Env {}'.format(env_name))

    if is_ple_game(env_name) or is_atari_env(env_name):
        if args['recurrent']:
            args['frame_history_len'] = 1
        else:
            args['frame_history_len'] = 4

    env = make_env(env_name, args)

    policy_net = a2c.get_policy_net(env, args)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args['learning_rate'])

    if args['recurrent']:
        init_hidden_unit = np.zeros(shape=(args['hidden_size']))
    else:
        init_hidden_unit = None

    agent = pg.A2CAgent(policy_net, policy_optimizer,
                        init_hidden_unit=init_hidden_unit,
                        nn_baseline=args['nn_baseline'],
                        lam=args['gae_lambda'],
                        value_coef=args['value_coef'])

    max_path_length = args['ep_len'] if args['ep_len'] > 0 else None

    pg.train(args['exp_name'], env, agent, args['n_iter'], args['discount'], args['batch_size'],
             max_path_length, logdir=None, seed=args['seed'])
