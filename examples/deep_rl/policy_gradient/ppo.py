"""
Test Vanilla PG on standard environment, where state is (ob_dim) and action is continuous/discrete

"""

import pprint

import numpy as np
import torch.optim

import torchlib.deep_rl.algorithm as algo
import torchlib.utils.random
from torchlib import deep_rl

if __name__ == '__main__':
    parser = algo.ppo.make_default_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    env_name = args['env_name']

    assert args['discount'] < 1.0, 'discount must be smaller than 1.0'
    torchlib.utils.random.set_global_seeds(args['seed'])

    print('Env {}'.format(args['env_name']))

    if deep_rl.envs.is_ple_game(env_name) or deep_rl.envs.is_atari_env(env_name):
        if args['recurrent']:
            args['frame_history_len'] = 1
        else:
            args['frame_history_len'] = 4
    else:
        args['frame_history_len'] = 1

    env = deep_rl.envs.make_env(args['env_name'], args)

    policy_net = algo.ppo.get_policy_net(env, args)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args['learning_rate'])

    if args['recurrent']:
        init_hidden_unit = np.zeros(shape=(args['hidden_size']))
    else:
        init_hidden_unit = None

    agent = algo.ppo.PPOAgent(policy_net, policy_optimizer,
                              init_hidden_unit=init_hidden_unit,
                              lam=args['gae_lambda'],
                              clip_param=args['clip_param'],
                              entropy_coef=args['entropy_coef'],
                              value_coef=args['value_coef'],
                              target_kl=args['target_kl'],
                              max_grad_norm=args['max_grad_norm'],
                              initial_state_mean=args['initial_state_mean'],
                              initial_state_std=args['initial_state_std'])

    checkpoint_path = 'checkpoint/{}_ppo.ckpt'.format(args['env_name'])

    max_path_length = args['ep_len'] if args['ep_len'] > 0 else None

    if args['test']:
        agent.load_checkpoint(checkpoint_path)
        deep_rl.test(env, agent, num_episode=args['n_iter'], render=True,
                     seed=args['seed'])

    else:
        agent.train(args['exp_name'], env, args['n_iter'], args['discount'], args['batch_size'],
                    max_path_length, logdir=None, seed=args['seed'], checkpoint_path=checkpoint_path)
