"""
Exploration strategies
"""
import numpy as np


class Exploration(object):
    def __call__(self, t):
        raise NotImplementedError


class EpsilonGreedy(Exploration):
    def __init__(self, epsilon=1.0, decay=1e-4, minimum=0.01):
        self.epsilon = epsilon
        self.decay = decay
        self.minimum = minimum

    def __call__(self, t):
        explore_p = self.minimum + (self.epsilon - self.minimum) * np.exp(-self.decay * t)
        return explore_p
