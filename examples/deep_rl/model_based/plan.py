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
    parser.add_argument('--num_iter', type=int, default=100)
    parser.add_argument('--num_init_random_rollouts', type=int, default=10)
    parser.add_argument('--max_rollout_length', type=int, default=1000)
    parser.add_argument('--num_on_policy_iters', type=int, default=10)
    parser.add_argument('--num_on_policy_rollouts', type=int, default=10)
    parser.add_argument('--training_epochs', type=int, default=60)
    parser.add_argument('--training_batch_size', type=int, default=128)
    parser.add_argument('--dataset_maxlen', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--fit_reward', action='store_true')
    parser.add_argument('--single_process', action='store_true')
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
    from torchlib.utils.random.sampler import UniformSampler, IntSampler

    model_based_wrapper = deep_rl.envs.wrappers.get_model_based_wrapper(args['env_name'])

    env = deep_rl.envs.make_env(args['env_name'], num_envs=None)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]

    dagger = args['dagger']

    if discrete:
        ac_dim = env.action_space.n
        dynamics_model = deep_rl.models.DiscreteMLPDynamics(state_dim=ob_dim, action_dim=ac_dim,
                                                            nn_size=args['nn_size'])
        action_sampler = IntSampler(low=ac_dim)
        if dagger:
            actor = deep_rl.models.ActorModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                               output_activation=None)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args['learning_rate'])
            policy = deep_rl.algorithm.model_based.DiscreteImitationPolicy(actor, actor_optimizer)

    else:
        ac_dim = env.action_space.shape[0]
        print('Action high: {}. Action low: {}'.format(env.action_space.high, env.action_space.low))
        dynamics_model = deep_rl.models.ContinuousMLPDynamics(state_dim=ob_dim, action_dim=ac_dim,
                                                              nn_size=args['nn_size'])
        action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)

        if dagger:
            actor = deep_rl.models.ActorModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                               output_activation=torch.tanh)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args['learning_rate'])
            policy = deep_rl.algorithm.model_based.policy.ContinuousImitationPolicy(actor, actor_optimizer)


    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=args['learning_rate'])

    if args['fit_reward']:
        env.cost_fn_batch = None

    model = deep_rl.algorithm.model_based.DeterministicWorldModel(dynamics_model=dynamics_model, optimizer=optimizer,
                                                                  cost_fn_batch=env.cost_fn_batch)

    planner = deep_rl.algorithm.model_based.planner.BestRandomActionPlanner(model=model,
                                                                            action_sampler=action_sampler,
                                                                            horizon=args['horizon'],
                                                                            num_random_action_selection=args[
                                                                                'num_actions'],
                                                                            gamma=args['gamma'])

    if dagger:
        agent = deep_rl.algorithm.model_based.agent.ModelBasedDAggerAgent(model=model, planner=planner, policy=policy,
                                                                          policy_data_size=args['dataset_maxlen'])
    else:
        agent = deep_rl.algorithm.model_based.agent.ModelBasedPlanAgent(model=model, planner=planner)

    single_env = env if args['single_process'] else None

    agent.train(env=single_env,
                dataset_maxlen=args['dataset_maxlen'],
                num_init_random_rollouts=args['num_init_random_rollouts'],
                max_rollout_length=args['max_rollout_length'],
                num_on_policy_iters=args['num_on_policy_iters'],
                num_on_policy_rollouts=args['num_on_policy_rollouts'],
                model_training_epochs=args['training_epochs'],
                training_batch_size=args['training_batch_size'],
                policy_training_epochs=args['training_epochs'],
                verbose=True,
                checkpoint_path=None)
