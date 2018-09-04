"""
Utilities to generate PyTorch random numbers
"""

import random

import numpy as np
import torch


def uniform_tensor(*shape, r1=0, r2=1):
    return (r2 - r1) * torch.rand(shape) + r1


def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
