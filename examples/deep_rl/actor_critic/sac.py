import pprint

import gym
import numpy as np
import torch.optim

import torchlib.deep_rl.actor_critic.sac as sac
from torchlib import deep_rl
from torchlib.common import device
from torchlib.deep_rl.envs import make_env

if __name__ == '__main__':
    parser = sac.make_default_parser()

    parser.add_argument('--nn_size', type=int, default=64)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', action='store_true')

    args = vars(parser.parse_args())
    pprint.pprint(args)

    env = make_env(args['env_name'], args)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    policy_net, q_network = sac.get_policy_net_q_network(env, args)

    learning_rate = args['learning_rate']

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

    automatic_entropy_tuning = not args['no_automatic_entropy_tuning']

    if automatic_entropy_tuning:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha_tensor = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha_tensor], lr=learning_rate)

    else:
        target_entropy = None
        log_alpha_tensor = None
        alpha_optimizer = None

    agent = sac.SoftActorCritic(policy_net=policy_net, policy_optimizer=policy_optimizer,
                                q_network=q_network, q_optimizer=q_optimizer,
                                discrete=discrete,
                                target_entropy=target_entropy,
                                log_alpha_tensor=log_alpha_tensor,
                                alpha_optimizer=alpha_optimizer,
                                tau=args['tau'], alpha=args['alpha'], discount=args['discount'])

    checkpoint_path = 'checkpoint/{}_SAC.ckpt'.format(args['env_name'])

    if not args['test']:
        sac.train(args['exp_name'], env, agent, args['n_epochs'], max_episode_length=args['max_episode_length'],
                  prefill_steps=args['prefill_steps'], epoch_length=args['epoch_length'],
                  replay_pool_size=args['replay_pool_size'], batch_size=args['batch_size'],
                  seed=args['seed'], checkpoint_path=checkpoint_path)
    else:
        agent.load_checkpoint(checkpoint_path=checkpoint_path)
        deep_rl.test(env, agent, num_episode=args['n_epochs'], render=args['render'], seed=args['seed'],
                     frame_history_len=args['frame_history_len'])
