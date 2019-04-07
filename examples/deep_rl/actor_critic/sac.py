import pprint

import numpy as np
import gym
import torch.optim
import torchlib.deep_rl.actor_critic.sac as sac
from torchlib import deep_rl
from torchlib.deep_rl.models.policy import ContinuousNNFeedForwardPolicy
from torchlib.deep_rl.models.value import DoubleCriticModule

__all__ = ['deep_rl']

if __name__ == '__main__':
    parser = sac.make_default_parser()

    parser.add_argument('--nn_size', type=int, default=64)

    args = vars(parser.parse_args())
    pprint.pprint(args)

    if args['env_name'].startswith('Roboschool'):
        import roboschool

        __all__.append('roboschool')

    env = gym.make(args['env_name'])

    print('Action space high', env.action_space.high)
    print('Action space low', env.action_space.low)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    policy_net = ContinuousNNFeedForwardPolicy(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

    q_network = DoubleCriticModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

    learning_rate = args['learning_rate']

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

    automatic_entropy_tuning = not args['no_automatic_entropy_tuning']

    if automatic_entropy_tuning:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha_tensor = torch.zeros(1, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha_tensor], lr=learning_rate)

    else:
        target_entropy = None
        log_alpha_tensor = None
        alpha_optimizer = None

    agent = sac.SoftActorCritic(policy_net=policy_net, policy_optimizer=policy_optimizer,
                                q_network=q_network, q_optimizer=q_optimizer,
                                target_entropy=target_entropy,
                                log_alpha_tensor=log_alpha_tensor,
                                alpha_optimizer=alpha_optimizer,
                                tau=args['tau'], alpha=args['alpha'], discount=args['discount'])

    sac.train(args['exp_name'], env, agent, args['n_epochs'], max_episode_length=args['max_episode_length'],
              prefill_steps=args['prefill_steps'], epoch_length=args['epoch_length'],
              replay_pool_size=args['replay_pool_size'], batch_size=args['batch_size'],
              seed=args['seed'])
