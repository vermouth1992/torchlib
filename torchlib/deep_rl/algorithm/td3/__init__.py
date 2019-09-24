import gym
from torchlib.deep_rl.models import ActorModule, DoubleCriticModule

from .agent import Agent


def add_args(parser):
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    parser.add_argument('--prefill_steps', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--epoch_length', type=int, default=5000)
    parser.add_argument('--replay_pool_size', type=int, default=1000000)
    parser.add_argument('--num_updates', type=int, default=2)
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_noise', type=float, default=0.2)
    parser.add_argument('--clip_noise', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--nn_size', type=int, default=64)
    return parser


def get_nets(env: gym.Env, args):
    actor_module = ActorModule(size=args['nn_size'],
                               state_dim=env.observation_space.low.shape[0],
                               action_dim=env.action_space.low.shape[0])

    critic_module = DoubleCriticModule(size=args['nn_size'], state_dim=env.observation_space.low.shape[0],
                                       action_dim=env.action_space.low.shape[0])

    return {
        'actor': actor_module,
        'critic': critic_module,
    }
