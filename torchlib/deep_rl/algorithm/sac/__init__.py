from torchlib import deep_rl

from .agent import Agent


def add_args(parser):
    # constructor arguments
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--no_automatic_alpha', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--epoch_length', type=int, default=5000)
    parser.add_argument('--prefill_steps', type=int, default=1000)
    parser.add_argument('--replay_pool_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--nn_size', '-s', type=int, default=64)

    parser.add_argument('--exp_name', type=str, default='sac')

    return parser


def get_nets(env, args):
    """ Return the policy network and q network

    Args:
        env: standard gym env instance
        args: arguments

    Returns: policy network, q network

    """
    discrete = deep_rl.envs.is_discrete(env)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    if len(env.observation_space.shape) == 1:
        # low dimensional environment
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        if discrete:
            from torchlib.deep_rl.models import CategoricalNNPolicy, DoubleQModule
            policy_net = CategoricalNNPolicy(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)
            q_network = DoubleQModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)
        else:
            from torchlib.deep_rl.models import TanhNormalNNPolicy, DoubleCriticModule

            policy_net = TanhNormalNNPolicy(nn_size=args['nn_size'], state_dim=ob_dim,
                                            action_dim=ac_dim)
            q_network = DoubleCriticModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

        return {
            'policy_net': policy_net,
            'q_network': q_network
        }

    elif len(env.observation_space.shape) == 3:
        if env.observation_space.shape[:2] == (84, 84):
            # atari env
            from torchlib.deep_rl.models import AtariPolicy, DoubleAtariQModule
            policy_net = AtariPolicy(num_channel=args['frame_history_len'],
                                     action_dim=env.action_space.n)
            q_network = DoubleAtariQModule(frame_history_len=args['frame_history_len'],
                                           action_dim=env.action_space.n)
            return {
                'policy_net': policy_net,
                'q_network': q_network
            }
        else:
            raise ValueError('Not a typical env. Please define custom network')

    else:
        raise ValueError('Not a typical env. Please define custom network')
