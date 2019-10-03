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
