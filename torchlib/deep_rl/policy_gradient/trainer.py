import datetime
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .a2c import A2CAgent
from .utils import sample_trajectories, pathlength


def train(exp, env, agent: A2CAgent, n_iter, gamma, min_timesteps_per_batch, max_path_length,
          logdir=None, seed=1996, checkpoint_path=None):
    # Set random seeds
    env.seed(seed)
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    total_timesteps = 0

    if logdir:
        writer = SummaryWriter(log_dir=os.path.join(logdir, exp))
    else:
        writer = None

    best_avg_return = None

    start_time = time.time()

    for itr in range(n_iter):
        paths, timesteps_this_batch = sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length)

        total_timesteps += timesteps_this_batch

        datasets = agent.construct_dataset(paths, gamma)
        agent.update_policy(datasets)

        print('-----------------------------------------------------------------------------------')
        print('Iteration {}/{} - Number of paths {} - Timesteps this batch {} - Total timesteps {}'.format(
            itr + 1,
            n_iter,
            len(paths),
            timesteps_this_batch,
            total_timesteps))

        # logger
        returns = [np.sum(path["reward"]) for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)

        if best_avg_return is None or avg_return > best_avg_return:
            best_avg_return = avg_return
            if checkpoint_path:
                print('Saving checkpoint to {}'.format(checkpoint_path))
                agent.save_checkpoint(checkpoint_path=checkpoint_path)

        if writer:
            writer.add_scalars('data/return', {'avg': avg_return,
                                               'std': std_return,
                                               'max': max_return,
                                               'min': min_return}, itr)
            writer.add_scalars('data/episode_length', {'avg': np.mean(ep_lengths),
                                                       'std': np.std(ep_lengths)}, itr)

        del datasets, paths

        time_elapse = datetime.timedelta(seconds=int(time.time() - start_time))

        print('Return {:.2f}±{:.2f} - Return range [{:.2f}, {:.2f}] - Best Avg Return {:.2f} - Time elapsed {}'.format(
            avg_return,
            std_return,
            min_return,
            max_return,
            best_avg_return,
            time_elapse))
