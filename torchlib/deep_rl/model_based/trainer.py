from gym import Env

from torchlib.deep_rl import RandomAgent
from .agent import VanillaAgent
from .utils import EpisodicDataset as Dataset, gather_rollouts


def train(env: Env, agent: VanillaAgent,
          dataset_maxlen=10000,
          num_init_random_rollouts=10,
          max_rollout_length=500,
          num_on_policy_iters=10,
          num_on_policy_rollouts=10,
          training_epochs=60,
          training_batch_size=512,
          verbose=True,
          checkpoint_path=None):
    # collect dataset using random policy
    random_policy = RandomAgent(env.action_space)
    dataset = Dataset(maxlen=dataset_maxlen)

    print('Gathering initial dataset...')
    initial_dataset = gather_rollouts(env, random_policy, num_init_random_rollouts, max_rollout_length)
    dataset.append(initial_dataset)

    # gather new rollouts using MPC and retrain dynamics model
    for num_iter in range(num_on_policy_iters):
        if verbose:
            print('On policy iteration {}/{}. Size of dataset: {}. Number of trajectories: {}'.format(
                num_iter + 1, num_on_policy_iters, len(dataset), dataset.num_trajectories))

        agent.set_statistics(dataset)

        agent.fit_dynamic_model(dataset=dataset, epoch=training_epochs, batch_size=training_batch_size,
                                verbose=verbose)
        on_policy_dataset = gather_rollouts(env, agent, num_on_policy_rollouts, max_rollout_length)

        # record on policy dataset statistics
        if verbose:
            stats = on_policy_dataset.log()
            strings = []
            for key, value in stats.items():
                strings.append(key + ": {:.4f}".format(value))
            strings = " - ".join(strings)
            print(strings)

        dataset.append(on_policy_dataset)

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)
