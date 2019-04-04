from torchlib.deep_rl.actor_critic.sac import SoftActorCritic, train, make_default_parser
import pprint
from torchlib import deep_rl
import gym

from torchlib.deep_rl.models.policy import ContinuousNNPolicy

__all__ = ['deep_rl']

if __name__ == '__main__':
    parser = make_default_parser()

    parser.add_argument('--nn_size', 64)

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

    policy_net = ContinuousNNPolicy(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                    recurrent=False, hidden_size=None)



