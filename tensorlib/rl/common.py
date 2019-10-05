from abc import ABC, abstractmethod

import gym
import numpy as np
from gym.spaces import Space
from tensorlib.utils.random import set_global_seeds


class BaseAgent(ABC):
    def predict(self, state):
        return self.predict_batch(np.expand_dims(state, axis=0))[0]

    @abstractmethod
    def predict_batch(self, states):
        raise NotImplementedError

    def reset(self):
        """
        This function is used for stateful agent such as recurrent agent.
        """
        pass

    @property
    def state_dict(self):
        return {}

    def load_state_dict(self, states):
        pass

    def save_checkpoint(self, checkpoint_path):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, action_space: Space):
        """

        Args:
            action_space: Must be batch action space
        """
        self.action_space = action_space

    def predict_batch(self, states):
        return self.action_space.sample()


def test(env: gym.Env, agent: BaseAgent, num_episode=100, frame_history_len=1, render=False, seed=1996):
    set_global_seeds(seed)
    env.seed(seed)
    reward_lst = []
    for i in range(num_episode):
        observation_lst = []
        done = False
        episode_reward = 0
        previous_observation = env.reset()
        agent.reset()
        observation_lst.append(previous_observation)
        for _ in range(frame_history_len - 1):
            if render:
                env.render()
            action = env.action_space.sample()
            previous_observation, reward, done, _ = env.step(action)
            observation_lst.append(previous_observation)
            episode_reward += reward
        while not done:
            if render:
                env.render()
            action = agent.predict(np.concatenate(observation_lst, axis=-1))
            previous_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            observation_lst.pop(0)
            observation_lst.append(previous_observation)
        print('Episode: {}/{}. Reward: {}'.format(i + 1, num_episode, episode_reward))
        reward_lst.append(episode_reward)
    print('Reward range [{}, {}]'.format(np.min(reward_lst), np.max(reward_lst)))
    print('Reward {}±{}'.format(np.mean(reward_lst), np.std(reward_lst)))

    env.close()
