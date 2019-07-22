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
    parser.add_argument('--training_epochs', type=int, default=60)
    parser.add_argument('--training_batch_size', type=int, default=128)
    parser.add_argument('--dataset_maxlen', type=int, default=10000)

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args = vars(args)

    import pprint

    pprint.pprint(args)

