import pprint
from itertools import chain

import gym
import torch.optim
import torchlib.deep_rl.actor_critic.sac as sac
from torchlib import deep_rl
from torchlib.deep_rl.models.policy import ContinuousNNPolicySAC
from torchlib.deep_rl.models.value import CriticModule, ValueModule

__all__ = ['deep_rl']

if __name__ == '__main__':
    parser = sac.make_default_parser()

    parser.add_argument('--nn_size', type=int, default=128)

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

    policy_net = ContinuousNNPolicySAC(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

    q_network = CriticModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)
    q_network2 = CriticModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

    value_network = ValueModule(size=args['nn_size'], state_dim=ob_dim)

    learning_rate = args['learning_rate']

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    q_optimizer = torch.optim.Adam(chain(q_network.parameters(), q_network2.parameters()), lr=learning_rate)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=learning_rate)

    agent = sac.SoftActorCritic(policy_net=policy_net, policy_optimizer=policy_optimizer,
                                q_network=q_network, q_network2=q_network2, q_optimizer=q_optimizer,
                                value_network=value_network, value_optimizer=value_optimizer,
                                tau=args['tau'], reparameterize=args['reparameterize'],
                                alpha=args['alpha'], discount=args['discount'])

    sac.train(args['exp_name'], env, agent, args['n_epochs'], max_episode_length=args['max_episode_length'],
              prefill_steps=args['prefill_steps'], epoch_length=args['epoch_length'],
              replay_pool_size=args['replay_pool_size'], batch_size=args['batch_size'],
              seed=args['seed'])
