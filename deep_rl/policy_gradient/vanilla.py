"""
Pytorch implementation of Vanilla PG. The code structure is adapted from UC Berkeley CS294-112.
Use various optimization techniques
1. Reward-to-go
2. Neural network baseline
3. Normalize advantage
4. Multiple threads to sample trajectory.
5. GAE-lambda
6. Multiple step update for PG
"""

import torch.nn as nn


class Agent(object):
    def __init__(self, policy_network: nn.Module, ):
        super(Agent, self).__init__()

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass


def train(exp_name, env_name, n_iter, ):
    pass
