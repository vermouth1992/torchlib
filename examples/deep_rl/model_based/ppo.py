"""
Fit world model using neural networks and train a policy network within the world model.
"""


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_init_random_rollouts', type=int, default=10)
    parser.add_argument('--max_rollout_length', type=int, default=500)
    parser.add_argument('--num_on_policy_iters', type=int, default=10)
    parser.add_argument('--num_on_policy_rollouts', type=int, default=10)
    parser.add_argument('--model_training_epochs', type=int, default=60)
    parser.add_argument('--policy_training_epochs', type=int, default=20)
    parser.add_argument('--training_batch_size', type=int, default=128)
    parser.add_argument('--dataset_maxlen', type=int, default=10000)
    # ppo agent arguments
    parser.add_argument('--recurrent', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--ppo_batch_size', type=int, default=1000)
    parser.add_argument('--ppo_ep_len', type=int, default=-1)
    parser.add_argument('--ppo_gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=1996)

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args = vars(args)

    import pprint
    import torchlib.deep_rl as deep_rl
    import torch.optim
    import gym

    pprint.pprint(args)

    # make env
    env = deep_rl.envs.make_env(args['env_name'], args)
    env = deep_rl.envs.wrappers.get_model_based_wrapper(args['env_name'])(env)

    # create ppo agent
    policy_net = deep_rl.algorithm.ppo.get_policy_net(env, args)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args['learning_rate'])
    ppo_agent = deep_rl.algorithm.ppo.PPOAgent(policy_net, policy_optimizer, None, lam=0.95)

    # create world model
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]

    if discrete:
        ac_dim = env.action_space.n
        dynamics_model = deep_rl.models.DiscreteMLPDynamics(state_dim=ob_dim, action_dim=ac_dim,
                                                            nn_size=args['nn_size'])
    else:
        ac_dim = env.action_space.shape[0]
        print('Action high: {}. Action low: {}'.format(env.action_space.high, env.action_space.low))
        dynamics_model = deep_rl.models.ContinuousMLPDynamics(state_dim=ob_dim, action_dim=ac_dim,
                                                              nn_size=args['nn_size'])

    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=args['learning_rate'])

    world_model = deep_rl.algorithm.model_based.DeterministicWorldModel(dynamics_model, optimizer)

    # create model-based agent
    agent = deep_rl.algorithm.model_based.ModelBasedPPOAgent(world_model, ppo_agent, env,
                                                             min_timesteps_per_batch=args['ppo_batch_size'],
                                                             max_path_length=args['ppo_ep_len'],
                                                             gamma=args['ppo_gamma'],
                                                             seed=args['seed'])

    # train
    agent.train(env=env,
                dataset_maxlen=args['dataset_maxlen'],
                num_init_random_rollouts=args['num_init_random_rollouts'],
                max_rollout_length=args['max_rollout_length'],
                num_on_policy_iters=args['num_on_policy_iters'],
                num_on_policy_rollouts=args['num_on_policy_rollouts'],
                model_training_epochs=args['model_training_epochs'],
                training_batch_size=args['training_batch_size'],
                policy_training_epochs=args['policy_training_epochs'],
                verbose=True,
                checkpoint_path=None)
