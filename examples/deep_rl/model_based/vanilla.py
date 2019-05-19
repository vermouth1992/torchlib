"""
Test Vanilla model-based RL
"""


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--num_random_actions', type=int, default=4096)
    parser.add_argument('--num_init_random_rollouts', type=int, default=10)
    parser.add_argument('--max_rollout_length', type=int, default=500)
    parser.add_argument('--num_on_policy_iters', type=int, default=10)
    parser.add_argument('--num_on_policy_rollouts', type=int, default=10)
    parser.add_argument('--training_epochs', type=int, default=60)
    parser.add_argument('--training_batch_size', type=int, default=128)
    return parser

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    args = vars(args)

    import pprint

    pprint.pprint(args)

    import gym
    import torch.optim
    from torchlib import deep_rl
    from torchlib.deep_rl.models.dynamics import MLPDynamics
    from torchlib.utils.random.sampler import UniformSampler
    import torchlib.deep_rl.model_based.vanilla as vanilla

    __all__ = ['deep_rl']

    if args['env_name'].startswith('Roboschool'):
        import roboschool

        __all__.append('roboschool')

    env = gym.make(args['env_name'])
    # discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Box), 'Only focus on continuous action space'
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    dynamics_model = MLPDynamics(state_dim=ob_dim, action_dim=ac_dim, nn_size=args['nn_size'])
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=args['learning_rate'])
    action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)

    agent = vanilla.Agent(dynamics_model=dynamics_model, optimizer=optimizer,
                          action_sampler=action_sampler, cost_fn=env.cost_fn,
                          horizon=args['horizon'], num_random_action_selection=args['num_random_actions'])

    vanilla.train(env, agent,
                  num_init_random_rollouts=args['num_init_random_rollouts'],
                  max_rollout_length=args['max_rollout_length'],
                  num_on_policy_iters=args['num_on_policy_iters'],
                  num_on_policy_rollouts=args['num_on_policy_rollouts'],
                  training_epochs=args['training_epochs'],
                  training_batch_size=args['training_batch_size'],
                  verbose=True,
                  checkpoint_path=None)
