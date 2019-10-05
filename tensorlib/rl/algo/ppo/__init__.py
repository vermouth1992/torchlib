import argparse

from .agent import Agent


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--lam', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--value_coef', type=float, default=0.5)

    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--num_updates', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--min_steps_per_batch', '-b', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.9, help='value mean/std moving ratio')

    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--policy', type=str, default='tanh_normal')
    parser.add_argument('--shared', action='store_true')

    parser.add_argument('--exp_name', type=str, default='ppo')


def get_nets(env, args):
    import gym
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    from tensorlib.rl.envs import is_atari_env, is_ple_game

    if len(env.observation_space.shape) == 1:
        # low dimensional environment
        if discrete:
            from tensorlib.rl.models.policy import CategoricalNNPolicyValue
            policy_net = CategoricalNNPolicyValue(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                                  shared=args['shared'])
        else:
            if args['policy'] == 'beta':
                from tensorlib.rl.models.policy import BetaNNPolicyValue
                policy_net = BetaNNPolicyValue(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                               shared=args['shared'])
            elif args['policy'] == 'normal':
                from tensorlib.rl.models.policy import NormalNNPolicyValue
                policy_net = NormalNNPolicyValue(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                                 shared=args['shared'])
            elif args['policy'] == 'tanh_normal':
                from tensorlib.rl.models.policy import TanhNormalNNPolicyValue
                policy_net = TanhNormalNNPolicyValue(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                                     shared=args['shared'])
            else:
                raise NotImplementedError

        return policy_net

    elif is_atari_env(env.spec.id) or is_ple_game(env.spec.id):
        if env.observation_space.shape[:2] == (84, 84):
            frame_history_len = env.observation_space.shape[-1]

            from tensorlib.rl.models.policy import AtariPolicy
            policy_net = AtariPolicy(num_channel=frame_history_len, action_dim=env.action_space.n)

            return policy_net
        else:
            raise ValueError('Not a typical env. Please define custom network')
    else:
        raise ValueError('Not a typical env. Please define custom network')
