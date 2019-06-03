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
    parser.add_argument('--num_actions', type=int, default=4096)
    parser.add_argument('--num_init_random_rollouts', type=int, default=10)
    parser.add_argument('--max_rollout_length', type=int, default=500)
    parser.add_argument('--num_on_policy_iters', type=int, default=10)
    parser.add_argument('--num_on_policy_rollouts', type=int, default=10)
    parser.add_argument('--training_epochs', type=int, default=60)
    parser.add_argument('--training_batch_size', type=int, default=128)
    parser.add_argument('--dataset_maxlen', type=int, default=10000)
    parser.add_argument('--planner', type=str, choices=['random', 'uct'], default='random')
    parser.add_argument('--dagger', action='store_true')
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
    from torchlib.deep_rl.models.dynamics import ContinuousMLPDynamics, DiscreteMLPDynamics
    from torchlib.deep_rl.model_based.model import DeterministicModel
    from torchlib.deep_rl.model_based.planner import BestRandomActionPlanner, UCTPlanner
    from torchlib.deep_rl.model_based.policy import DiscretePolicy, ContinuousPolicy
    from torchlib.deep_rl.models.policy import ActorModule
    from torchlib.utils.random.sampler import UniformSampler, IntSampler
    from torchlib.deep_rl.model_based.agent import VanillaAgent, DAggerAgent
    import torchlib.deep_rl.model_based.trainer as trainer

    __all__ = ['deep_rl']

    if args['env_name'].startswith('Roboschool'):
        import roboschool

        __all__.append('roboschool')

    env = gym.make(args['env_name'])
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]

    dagger = args['dagger']

    if discrete:
        ac_dim = env.action_space.n
        dynamics_model = DiscreteMLPDynamics(state_dim=ob_dim, action_dim=ac_dim, nn_size=args['nn_size'])
        action_sampler = IntSampler(low=ac_dim)
        if dagger:
            actor = ActorModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim, output_activation=None)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args['learning_rate'])
            policy = DiscretePolicy(actor, actor_optimizer)
    else:
        ac_dim = env.action_space.shape[0]
        print('Action high: {}. Action low: {}'.format(env.action_space.high, env.action_space.low))
        dynamics_model = ContinuousMLPDynamics(state_dim=ob_dim, action_dim=ac_dim, nn_size=args['nn_size'])
        action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)

        if dagger:
            actor = ActorModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim, output_activation=torch.tanh)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args['learning_rate'])
            policy = ContinuousPolicy(actor, actor_optimizer)

    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=args['learning_rate'])

    model = DeterministicModel(dynamics_model=dynamics_model, optimizer=optimizer)

    if args['planner'] == 'random':
        planner = BestRandomActionPlanner(model=model, action_sampler=action_sampler, cost_fn=env.cost_fn,
                                          horizon=args['horizon'],
                                          num_random_action_selection=args['num_actions'])
    elif args['planner'] == 'uct':
        planner = UCTPlanner(model, action_sampler, env.cost_fn, horizon=args['horizon'],
                             num_reads=args['num_actions'])
    else:
        raise ValueError('Unknown planner {}'.format(args['planner']))

    if dagger:
        agent = DAggerAgent(model=model, planner=planner, policy=policy, policy_data_size=args['dataset_maxlen'])
    else:
        agent = VanillaAgent(model=model, planner=planner)

    trainer.train(env, agent,
                  dataset_maxlen=args['dataset_maxlen'],
                  num_init_random_rollouts=args['num_init_random_rollouts'],
                  max_rollout_length=args['max_rollout_length'],
                  num_on_policy_iters=args['num_on_policy_iters'],
                  num_on_policy_rollouts=args['num_on_policy_rollouts'],
                  training_epochs=args['training_epochs'],
                  training_batch_size=args['training_batch_size'],
                  verbose=True,
                  checkpoint_path=None)
