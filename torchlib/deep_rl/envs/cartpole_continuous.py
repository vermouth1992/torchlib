"""
Continuous cartpole
"""

import numpy as np
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv

from .model_based import ModelBasedEnv


class CartPoleCost(CartPoleEnv, ModelBasedEnv):
    def cost_fn(self, states, actions, next_states):
        """ Use next_states to determine whether it is done. If it is done, set cost to be 1. Otherwise, 0 """
        x = next_states[:, 0]
        theta = next_states[:, 2]
        x_done = np.logical_or(x < -self.x_threshold, x > self.x_threshold)
        theta_done = np.logical_or(theta < -self.theta_threshold_radians, theta > self.theta_threshold_radians)
        done = np.logical_or(x_done, theta_done).astype(np.int)
        return done


class CartPoleContinuous(CartPoleEnv):
    def __init__(self):
        super(CartPoleContinuous, self).__init__()
        self.new_action_space = spaces.Box(-1, 1, shape=(1,))
        self.raw_action_space = spaces.Discrete(2)
        self.action_space = self.new_action_space

    def step(self, action):
        action = np.clip(action, a_min=-1, a_max=1)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if action[0] > 0:
            action = 1
        else:
            action = 0

        self.action_space = self.raw_action_space
        result = super(CartPoleContinuous, self).step(action)
        self.action_space = self.new_action_space
        return result
