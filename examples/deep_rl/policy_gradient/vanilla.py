"""
Test Vanilla PG on standard environment, where state is (ob_dim) and action is continuous/discrete

"""

import gym.spaces
import torch
import torch.nn as nn
import torchlib.deep_rl
import torchlib.deep_rl.policy_gradient.vanilla as vanilla_pg
from torchlib.common import FloatTensor, enable_cuda
from torchlib.deep_rl.policy_gradient.vanilla import Agent


class PolicyDiscrete(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim):
        assert action_dim > 1, 'Action dim must be greater than 1. Got {}'.format(action_dim)
        super(PolicyDiscrete, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model.forward(state)


class PolicyContinuous(nn.Module):
    def __init__(self, nn_size, state_dim, action_dim):
        super(PolicyContinuous, self).__init__()
        self.logstd = torch.randn([action_dim, action_dim], requires_grad=True).type(FloatTensor)
        self.model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        mean = self.model.forward(state)
        return mean, self.logstd


class Baseline(nn.Module):
    def __init__(self, nn_size, state_dim):
        super(Baseline, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, 1),
        )

    def forward(self, state):
        return self.model.forward(state).squeeze(dim=-1)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    max_path_length = args.ep_len if args.ep_len > 0 else None

    if args.env_name.startswith('Roboschool'):
        pass

    env = gym.make(args.env_name)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    print('Env {}'.format(args.env_name))
    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    if discrete:
        policy_net = PolicyDiscrete(args.nn_size, ob_dim, ac_dim)
    else:
        policy_net = PolicyContinuous(args.nn_size, ob_dim, ac_dim)

    if enable_cuda:
        policy_net.cuda()

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args.learning_rate)

    if args.nn_baseline:
        baseline_net = Baseline(args.nn_size, ob_dim)
        baseline_optimizer = torch.optim.Adam(baseline_net.parameters(), args.learning_rate)
        if enable_cuda:
            baseline_net.cuda()
    else:
        baseline_net = None
        baseline_optimizer = None

    agent = Agent(policy_net, policy_optimizer, discrete, baseline_net, baseline_optimizer)

    vanilla_pg.train(args.exp_name, env, agent, args.n_iter, args.discount, args.batch_size, max_path_length,
                     logdir='runs', seed=args.seed)
