"""
Test Vanilla PG on standard environment, where state is (ob_dim) and action is continuous/discrete

"""

import pprint

import numpy as np
import torch.optim

import torchlib.deep_rl.policy_gradient as pg
import torchlib.deep_rl.policy_gradient.ppo as ppo
from torchlib import deep_rl
from torchlib.deep_rl.envs import make_env
from torchlib.utils.random import set_global_seeds

if __name__ == '__main__':
    parser = ppo.make_default_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    assert args['discount'] < 1.0, 'discount must be smaller than 1.0'

    set_global_seeds(args['seed'])

    env = make_env(args['env_name'], args)

    policy_net = ppo.get_policy_net(env, args)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args['learning_rate'])

    if args['recurrent']:
        init_hidden_unit = np.zeros(shape=(args['hidden_size']))
    else:
        init_hidden_unit = None

    agent = pg.PPOAgent(policy_net, policy_optimizer,
                        init_hidden_unit=init_hidden_unit,
                        lam=args['gae_lambda'],
                        clip_param=args['clip_param'],
                        entropy_coef=args['entropy_coef'], value_coef=args['value_coef'])

    checkpoint_path = 'checkpoint/{}_ppo.ckpt'.format(args['env_name'])

    max_path_length = args['ep_len'] if args['ep_len'] > 0 else None

    if args['test']:
        agent.load_checkpoint(checkpoint_path)
        deep_rl.test(env, agent, num_episode=args['n_iter'], frame_history_len=args['frame_history_len'], render=False,
                     seed=args['seed'])

    else:
        pg.train(args['exp_name'], env, agent, args['n_iter'], args['discount'], args['batch_size'],
                 max_path_length, logdir=None, seed=args['seed'], checkpoint_path=checkpoint_path)
