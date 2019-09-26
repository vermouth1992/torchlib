"""
Common utilities to implement policy gradient algorithms
"""

from collections import namedtuple, deque

import numpy as np
from scipy import signal
from torchlib.dataset.utils import create_data_loader
from torchlib.deep_rl.utils.replay.replay import ReplayBuffer
from torchlib.deep_rl.utils.replay.sampler import Sampler
from torchlib.utils.math import unnormalize, normalize

Trajectory = namedtuple('Trajectory', ('state', 'action', 'reward_to_go', 'advantage', 'old_log_prob'))


class PPOReplayBuffer(ReplayBuffer):
    def __init__(self, gamma, lam, policy, alpha=0.9):
        """

        Args:
            gamma: discount factor
            lam: generalized advantage estimation
            policy: PPO policy
            alpha: value moving average ratio
        """
        super(PPOReplayBuffer, self).__init__(None, None, None, None, None)
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.policy = policy

    def _initialize(self):
        self.memory = deque()
        self.running_value_mean = 0.
        self.running_value_std = 0.

    def clear(self):
        self._size = 0
        self.memory.clear()

    def _finish_trajectory(self, states, actions, rewards, last_value):
        """Compute path accessory information including (reward_to_go, old_log_prob, advantage)

        Returns:

        """
        predicted_state_values = self.policy.predict_state_value_batch(states)
        predicted_state_values = unnormalize(predicted_state_values, self.running_value_mean, self.running_value_std)

        rewards_last_state = np.append(rewards, last_value)
        predicted_state_values = np.append(predicted_state_values, last_value)

        # Used for fit value function
        reward_to_go = discount(rewards_last_state, self.gamma).astype(np.float32)[:-1]

        temporal_difference = rewards + predicted_state_values[1:] * self.gamma - predicted_state_values[:-1]
        # calculate reward-to-go
        gae = discount(temporal_difference, self.gamma * self.lam).astype(np.float32)

        old_log_prob = self.policy.predict_log_prob_batch(states, actions)

        return reward_to_go, gae, old_log_prob

    def add_trajectory(self, states, actions, rewards, last_value):
        """If last_state is not None, this trajectory is truncated.

        Args:
            states: (T, ob_dim)
            actions: （T, ac_dim)
            rewards: (T,)
            last_state: (ob_dim)

        Returns:

        """
        reward_to_go, gae, old_log_prob = self._finish_trajectory(states, actions, rewards, last_value)
        self.memory.append(Trajectory(
            state=states,
            action=actions,
            reward_to_go=reward_to_go,
            advantage=gae,
            old_log_prob=old_log_prob
        ))

        self._size += actions.shape[0]

    def random_iterator(self, batch_size):
        """Create an iterator of all the dataset and update value mean and std


        Args:
            batch_size:

        Returns:

        """
        states = np.concatenate([trajectory.state for trajectory in self.memory], axis=0)
        actions = np.concatenate([trajectory.action for trajectory in self.memory], axis=0)
        reward_to_go = np.concatenate([trajectory.reward_to_go for trajectory in self.memory], axis=0)
        gaes = np.concatenate([trajectory.advantage for trajectory in self.memory], axis=0)
        old_log_prob = np.concatenate([trajectory.old_log_prob for trajectory in self.memory], axis=0)

        value_mean, value_std = np.mean(reward_to_go), np.std(reward_to_go)
        reward_to_go = normalize(reward_to_go, value_mean, value_std)

        self.running_value_mean = self.running_value_mean * self.alpha + value_mean * (1 - self.alpha)
        self.running_value_std = self.running_value_std * self.alpha + value_std * (1 - self.alpha)

        gaes = normalize(gaes, np.mean(gaes), np.std(gaes))

        batch_size = min(batch_size, states.shape[0])

        data_loader = create_data_loader((states, actions, reward_to_go, gaes, old_log_prob),
                                         batch_size=batch_size, shuffle=True, drop_last=True)

        return data_loader


class PPOSampler(Sampler):
    def __init__(self, min_steps_per_batch, logger=None):
        super(PPOSampler, self).__init__()
        self.min_steps_per_batch = min_steps_per_batch
        self.logger = logger

    def sample_trajectories(self, policy=None):
        obs_lst = []
        action_lst = []
        reward_lst = []
        done_lst = []

        policy = self.policy if policy is None else policy
        obs = self.env.reset()
        for _ in range(self.min_steps_per_batch // obs.shape[0]):
            action = policy.predict_batch(obs)
            obs_lst.append(obs)
            action_lst.append(action)

            obs, rewards, dones, infos = self.env.step(action)

            reward_lst.append(rewards)
            done_lst.append(dones)

        # compute last state value for the last trajectory in each environment
        last_state_lst = obs
        last_value_lst = self.policy.predict_state_value_batch(last_state_lst)
        last_value_lst = unnormalize(last_value_lst, self.pool.running_value_mean, self.pool.running_value_std)

        obs_lst = np.stack(obs_lst, axis=1)
        action_lst = np.stack(action_lst, axis=1)
        reward_lst = np.stack(reward_lst, axis=1)
        done_lst = np.stack(done_lst, axis=1)

        # separate trajectories and add to pool
        for i in range(self.env.num_envs):
            done_index = np.where(done_lst[i])[0] + 1
            if done_lst[i][-1] == True:
                done_index = done_index[:-1]  # ignore the last one
                last_value = 0.
            else:
                last_value = last_value_lst[i]

            sub_obs_lst = np.split(obs_lst[i], done_index)
            sub_action_lst = np.split(action_lst[i], done_index)
            sub_reward_lst = np.split(reward_lst[i], done_index)
            sub_last_value_lst = [0.] * (len(sub_obs_lst) - 1) + [last_value]

            for j in range(len(sub_obs_lst)):
                self.pool.add_trajectory(states=sub_obs_lst[j],
                                         actions=sub_action_lst[j],
                                         rewards=sub_reward_lst[j],
                                         last_value=sub_last_value_lst[j])
                if self.logger:
                    self.logger.store(EpReward=np.sum(sub_reward_lst[j]) + sub_last_value_lst[j])
                    self.logger.store(EpLength=sub_obs_lst[j].shape[0])


def discount(x, gamma):
    return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]
