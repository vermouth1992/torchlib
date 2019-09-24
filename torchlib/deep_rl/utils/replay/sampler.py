"""
Sample class for various RL algorithms. Typical RL algorithms contains two parts:
1. Sample trajectories or transitions and add to the buffer
2. Use the buffer to update the policy or value functions
"""

class Sampler(object):
    def __init__(self, max_episode_length, prefill_steps):
        self._max_episode_length = max_episode_length
        self._prefill_steps = prefill_steps

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

        for _ in range(self._prefill_steps):
            self.sample()

    def set_policy(self, policy):
        self.policy = policy

    def sample(self, policy=None):
        raise NotImplementedError

    def sample_trajectories(self, policy=None):
        raise NotImplementedError

    def random_batch(self, batch_size):
        return self.pool.random_batch(batch_size)

    def close(self):
        self.env.close()


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._episode_length = 0
        self._episode_return = 0
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._episode_rewards = []

    def sample(self, policy=None):
        policy = self.policy if policy is None else policy
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = policy.predict(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._episode_length += 1
        self._episode_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observations=self._current_observation,
            actions=action,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation)

        if terminal or self._episode_length >= self._max_episode_length:
            self._current_observation = self.env.reset()
            self._episode_length = 0
            self._episode_rewards.append(self._episode_return)
            self._episode_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def get_total_steps(self):
        return self._total_samples

    def get_episode_rewards(self):
        return self._episode_rewards.copy()
