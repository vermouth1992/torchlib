"""
Common utilities to implement policy gradient algorithms
"""

import numpy as np
import torch
from scipy import signal

from torchlib.common import FloatTensor, convert_numpy_to_tensor
from torchlib.dataset.utils import create_data_loader


def discount(x, gamma):
    return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def pathlength(path):
    return len(path["reward"])


def compute_last_state_value(path, policy_net, mean, std):
    if path['mask'][-1] == 1:
        with torch.no_grad():
            last_obs = convert_numpy_to_tensor(
                np.expand_dims(path['last_obs'], axis=0)).type(FloatTensor)
            last_hidden = convert_numpy_to_tensor(
                np.expand_dims(path['last_hidden'], axis=0)).type(FloatTensor)
            last_value = policy_net.forward(last_obs, last_hidden)[-1].cpu().numpy()[0]
        last_value = last_value * std + mean
    else:
        last_value = 0.
    return last_value


def compute_sum_of_rewards(paths, gamma, policy_net, mean, std):
    # calculate sum of reward
    rewards = []
    for path in paths:
        # compute last state value
        last_state_value = compute_last_state_value(path, policy_net, mean, std)
        # calculate reward-to-go
        path['reward'][-1] += last_state_value * gamma
        current_rewards = discount(path['reward'], gamma).tolist()
        rewards.extend(current_rewards)
    rewards = np.array(rewards, dtype=np.float32)
    return rewards


def compute_gae(paths, gamma, policy_net, lam, mean, std):
    gaes = []
    for path in paths:
        with torch.no_grad():
            observation = torch.from_numpy(path['observation'])
            hidden = torch.from_numpy(path['hidden'])
            data_loader = create_data_loader((observation, hidden), batch_size=32, shuffle=False, drop_last=False)
            values = []
            for obs, hid in data_loader:
                obs = obs.type(FloatTensor)
                hid = hid.type(FloatTensor)
                values.append(policy_net.forward(obs, hid)[-1])
            values = torch.cat(values, dim=0).cpu().numpy()
        values = values * std + mean

        # last_value = path['last_value']

        # add the value of last obs for truncated trajectory
        temporal_difference = path['reward'] + np.append(values[1:], 0) * gamma - values
        # calculate reward-to-go
        gae = discount(temporal_difference, gamma * lam).tolist()
        gaes.extend(gae)

    gaes = np.array(gaes, dtype=np.float32)
    return gaes


def sample_trajectory(agent, env, max_path_length):
    # this function should not participate in the computation graph
    ob = env.reset()
    agent.reset()
    actions, rewards, obs, hiddens, masks = [], [], [], [], []
    steps = 0
    while True:
        if ob.dtype == np.float:
            ob = ob.astype(np.float32)

        obs.append(ob)
        hiddens.append(agent.get_hidden_unit())

        ac = agent.predict(ob)

        if isinstance(ac, np.ndarray) and ac.dtype == np.float:
            ac = ac.astype(np.float32)

        actions.append(ac)

        ob, rew, done, _ = env.step(ac)
        rewards.append(rew)
        masks.append(int(not done))  # if done, mask is 0. Otherwise, 1.
        steps += 1
        if done or steps >= max_path_length:
            break

    if ob.dtype == np.float:
        ob = ob.astype(np.float32)

    path = {"actions": actions,
            "reward": rewards,
            "observation": np.array(obs),
            "hidden": np.array(hiddens),
            "mask": np.array(masks),
            'last_obs': ob,
            'last_hidden': agent.get_hidden_unit(),
            }
    return path


def sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length):
    timesteps_this_batch = 0
    paths = []
    while True:
        path = sample_trajectory(agent, env, max_path_length)
        paths.append(path)
        timesteps_this_batch += pathlength(path)
        if timesteps_this_batch >= min_timesteps_per_batch:
            break
    return paths, timesteps_this_batch
