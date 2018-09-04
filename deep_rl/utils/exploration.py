"""
Exploration strategies
"""
import numpy as np


class Exploration(object):
    def __call__(self, global_step, action):
        raise NotImplementedError


class EpsilonGreedy(Exploration):
    def __init__(self, epsilon=1.0, decay=1e-4, minimum=0.01, sampler=None):
        self.epsilon = epsilon
        self.decay = decay
        self.minimum = minimum
        self.sampler = sampler

    def __call__(self, global_step, action):
        explore_p = self.minimum + (self.epsilon - self.minimum) * np.exp(-self.decay * global_step)
        if np.random.rand() < explore_p:
            action_take = self.sampler.sample()
        else:
            action_take = action
        return action_take
