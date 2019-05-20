"""
Continuous cartpole
"""

import numpy as np
import torch
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv

from torchlib.common import LongTensor
from .model_based import ModelBasedEnv


class CartPoleCost(CartPoleEnv, ModelBasedEnv):
    def cost_fn(self, states, actions, next_states):
        return self.cost_fn_v2(states, actions, next_states)

    def cost_fn_v1(self, states, actions, next_states):
        """ Use next_states to determine whether it is done. If it is done, set cost to be 1. Otherwise, 0 """
        x = next_states[:, 0]
        theta = next_states[:, 2]
        if isinstance(states, torch.Tensor):
            x_done = (x < -self.x_threshold) | (x > self.x_threshold)
            theta_done = (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)
            done = x_done | theta_done
            done = done.type(LongTensor)
        elif isinstance(states, np.ndarray):
            x_done = np.logical_or(x < -self.x_threshold, x > self.x_threshold)
            theta_done = np.logical_or(theta < -self.theta_threshold_radians, theta > self.theta_threshold_radians)
            done = np.logical_or(x_done, theta_done).astype(np.int)
        else:
            raise ValueError('Unknown data type {}'.format(type(states)))
        return done

    def cost_fn_v2(self, states, actions, next_states):
        """ Use the absolute value of x and theta. The goal is to maintain them around zero. """
        x = next_states[:, 0]
        theta = next_states[:, 2]
        if isinstance(states, torch.Tensor):
            abs = torch.abs
        elif isinstance(states, np.ndarray):
            abs = np.abs
        else:
            raise ValueError('Unknown data type {}'.format(type(states)))

        return abs(x) / self.x_threshold + abs(theta) / self.theta_threshold_radians


class CartPoleContinuous(CartPoleCost):
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
