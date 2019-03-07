"""
Common utilities to implement policy gradient algorithms
"""

import numpy as np
import torch
from scipy import signal

from torchlib.common import FloatTensor


def discount(x, gamma):
    return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def pathlength(path):
    return len(path["reward"])


def compute_sum_of_rewards(paths, gamma):
    # calculate sum of reward
    rewards = []
    for path in paths:
        # calculate reward-to-go
        current_rewards = discount(path['reward'], gamma).tolist()
        rewards.extend(current_rewards)
    rewards = np.array(rewards)
    return rewards


def compute_gae(paths, gamma, policy_net, lam, mean, std):
    gaes = []
    for path in paths:
        with torch.no_grad():
            observation = torch.from_numpy(path['observation']).type(FloatTensor)
            hidden = torch.from_numpy(path['hidden']).type(FloatTensor)
            values = policy_net.forward(observation, hidden)[-1].cpu().numpy()
        values = values * std + mean
        temporal_difference = path['reward'] + np.append(values[1:] * gamma, 0) - values
        # calculate reward-to-go
        gae = discount(temporal_difference, gamma * lam).tolist()
        gaes.extend(gae)
    gaes = np.array(gaes)
    return gaes


def sample_trajectory(agent, env, max_path_length):
    # this function should not participate in the computation graph
    ob = env.reset()
    agent.reset()
    actions, rewards, obs, hiddens = [], [], [], []
    steps = 0
    while True:
        # distribution = agent.get_action_distribution(np.array([ob]))
        # # ac and log_prob are nodes on computational graph
        # ac = distribution.sample(torch.Size([1]))
        #
        # log_prob.append(distribution.log_prob(ac))

        obs.append(ob)
        hiddens.append(agent.get_hidden_unit())

        ac = agent.predict(ob)
        actions.append(ac)

        ob, rew, done, _ = env.step(ac)
        rewards.append(rew)
        steps += 1
        if done or steps > max_path_length:
            break
    path = {"actions": actions,
            "reward": rewards,
            "observation": np.array(obs),
            "hidden": np.array(hiddens)}
    return path


def sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length):
    timesteps_this_batch = 0
    paths = []
    while True:
        path = sample_trajectory(agent, env, max_path_length)
        paths.append(path)
        timesteps_this_batch += pathlength(path)
        if timesteps_this_batch > min_timesteps_per_batch:
            break
    return paths, timesteps_this_batch
