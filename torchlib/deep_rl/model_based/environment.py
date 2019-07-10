"""
Virtual Environment for training policy using model-free approach
"""

from collections import deque

import gym
import numpy as np

from torchlib.deep_rl.envs.model_based import ModelBasedEnv
from .model import Model


class VirtualEnv(gym.Env):
    def __init__(self, model: Model, real_env: ModelBasedEnv, max_initial_states):
        """ Virtual Environment. We only consider environment with pre-defined cost function.
        We will extend to learn reward function in the end.

        Args:
            model: model to predict next states given previous states and
            real_env: real environment
            max_initial_states:
        """
        super(VirtualEnv, self).__init__()
        self.action_space = real_env.action_space
        self.observation_space = real_env.observation_space
        self.reward_range = real_env.reward_range
        self.cost_fn = real_env.cost_fn
        self.model = model
        self.initial_states = deque(maxlen=max_initial_states)
        self.current_state = None

    def reset(self):
        self.current_state = np.random.choice(self.initial_states)
        return self.current_state

    def step(self, action):
        next_state = self.model.predict_next_state(self.current_state, action)
        reward = -self.cost_fn(np.expand_dims(self.current_state, axis=0),
                               np.expand_dims(action, axis=0),
                               np.expand_dims(next_state, axis=0))[0]
        self.current_state = next_state
        return self.current_state, reward, False, {}

    def seed(self, seed=None):
        pass

    def add_initial_state(self, initial_state):
        self.initial_states.append(initial_state)
