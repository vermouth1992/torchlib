"""
Utilities to generate PyTorch random numbers
"""

import torch


def uniform_tensor(*shape, r1=0, r2=1):
    return (r2 - r1) * torch.rand(shape) + r1
