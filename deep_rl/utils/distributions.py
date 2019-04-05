"""
Rewrite Pytorch builtin distribution function to favor policy gradient
1. Normal distribution with multiple mean and std as a single distribution
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal

from torch.distributions import Normal


class FixedNormal(Normal):
    def log_prob(self, value):
        result = super(FixedNormal, self).log_prob(value)
        return result.sum(-1)

    def entropy(self):
        return super(FixedNormal, self).entropy().sum(-1)


class FixedNormalTanh(FixedNormal):
    def log_prob(self, value):
        out = super(FixedNormalTanh, self).log_prob(value=value)
        out -= self._squash_correction(value)
        return out

    def _squash_correction(self, value):
        ### Problem 2.B
        ### YOUR CODE HERE
        return torch.sum(np.log(4.) + 2. * value - 2 * F.softplus(2. * value), dim=1)


class MultivariateNormalDiag(LowRankMultivariateNormal):
    def __init__(self, loc, scale):
        if loc.dim() == 1:
            cov_factor = torch.zeros(loc.shape[0], 1)
        elif loc.dim() == 2:
            cov_factor = torch.zeros(loc.shape[0], loc.shape[1], 1)
        else:
            raise ValueError('Unsupported loc shape {}'.format(loc.shape))
        super(MultivariateNormalDiag, self).__init__(loc=loc, cov_factor=cov_factor, cov_diag=scale)


class MultivariateNormalDiagTanh(MultivariateNormalDiag):
    """
    Add tanh function to low rank multivariate normal distribution
    """

    def log_prob(self, value):
        out = super(MultivariateNormalDiagTanh, self).log_prob(value=value)
        out -= self._squash_correction(value)
        return out

    def sample(self, sample_shape=torch.Size()):
        data = super(MultivariateNormalDiagTanh, self).sample(sample_shape=sample_shape)
        return torch.tanh(data)

    def rsample(self, sample_shape=torch.Size()):
        data = super(MultivariateNormalDiagTanh, self).rsample(sample_shape=sample_shape)
        return torch.tanh(data)

    def _squash_correction(self, value):
        ### Problem 2.B
        ### YOUR CODE HERE
        return torch.sum(np.log(4.) + 2. * value - 2 * F.softplus(2. * value), dim=1)
