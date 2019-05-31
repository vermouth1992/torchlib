import numpy as np
import torch
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv

from .model_based import ModelBasedEnv


class PendulumEnvCost(PendulumEnv, ModelBasedEnv):
    def __init__(self):
        super(PendulumEnvCost, self).__init__()
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        return super(PendulumEnvCost, self).step(u=action * self.max_torque)

    def cost_fn(self, states, actions, next_states):
        """ The cost function is exactly how the original env calculates rewards """
        if isinstance(states, torch.Tensor):
            atan2 = torch.atan2
        elif isinstance(states, np.ndarray):
            atan2 = np.arctan2
        else:
            raise ValueError('Unknown data type {}'.format(type(states)))

        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = atan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs
