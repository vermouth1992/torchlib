# check cuda for torch
import numpy as np
import torch

from . import enable_cuda

__all__ = ['enable_cuda', 'FloatTensor', 'LongTensor', 'map_location', 'eps']

FloatTensor = torch.cuda.FloatTensor if enable_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if enable_cuda else torch.LongTensor
map_location = None if enable_cuda else 'cpu'
eps = np.finfo(np.float32).eps.item()
