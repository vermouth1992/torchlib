"""
Low dimension control using TD3
"""

import pprint

import torch.optim
import torchlib.deep_rl.algorithm as rl_algorithm
from torchlib import deep_rl

if __name__ == '__main__':
    parser = rl_algorithm.td3.make_default_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    env_name = args['env_name']
    env = deep_rl.envs.make_env(env_name, args)
    env = deep_rl.envs.wrappers.ObservationDTypeWrapper(env)

    actor_module, critic_module = rl_algorithm.td3.get_default_actor_critic(env, args)
    actor_module_optimizer = torch.optim.Adam(actor_module.parameters(), lr=args['learning_rate'])
    critic_module_optimizer = torch.optim.Adam(critic_module.parameters(), lr=args['learning_rate'])

    agent = rl_algorithm.td3.TD3(actor_module, actor_module_optimizer,
                                 critic_module, critic_module_optimizer)

    agent.train(env, actor_noise=None, seed=args['seed'], start_steps=args['start_steps'],
                total_steps=args['total_steps'], replay_size=args['replay_size'], replay_buffer=None,
                num_updates=args['num_updates'], policy_freq=args['policy_freq'], batch_size=args['batch_size'],
                target_noise=args['target_noise'], clip_noise=args['clip_noise'], tau=args['tau'],
                gamma=args['gamma'], log_freq=args['log_freq'])
