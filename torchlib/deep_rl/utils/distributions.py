"""
Rewrite Pytorch builtin distribution function to favor policy gradient
1. Normal distribution with multiple mean and std as a single distribution
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Distribution, Independent

from torchlib.common import eps


class FixedNormal(Distribution):
    """
    Treat multiple normal distribution as a single distribution.
    The same as MultivariateNormalDiag in tensorflow.
    """

    def __init__(self, loc, scale, validate_args=None):
        super(FixedNormal, self).__init__()
        self.loc = loc
        self.scale = scale
        self.normal = Independent(Normal(loc=loc, scale=scale, validate_args=validate_args), 1,
                                  validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self.normal.sample(sample_shape=sample_shape).detach()

    def rsample(self, sample_shape=torch.Size()):
        return self.normal.rsample(sample_shape=sample_shape)

    def log_prob(self, value):
        return self.normal.log_prob(value)

    def entropy(self):
        return self.normal.entropy()

    @property
    def mean(self):
        return self.normal.mean

    @property
    def stddev(self):
        return self.normal.stddev


class FixedNormalTanh(FixedNormal):
    def sample(self, sample_shape=torch.Size(), return_raw_value=False):
        out = super(FixedNormalTanh, self).sample(sample_shape=sample_shape)
        if return_raw_value:
            return torch.tanh(out), out
        else:
            return torch.tanh(out)

    def rsample(self, sample_shape=torch.Size(), return_raw_value=False):
        out = super(FixedNormalTanh, self).rsample(sample_shape=sample_shape)
        if return_raw_value:
            return torch.tanh(out), out
        else:
            return torch.tanh(out)

    def log_prob(self, value, is_raw_value=False):
        """

        Args:
            value: We assume the value is the one after tanh.

        Returns:

        """
        if not is_raw_value:
            raw_value = 0.5 * (torch.log(1. + value + eps) - torch.log(1. - value + eps))
        else:
            raw_value = value
        out = super(FixedNormalTanh, self).log_prob(value=raw_value)
        if not is_raw_value:
            term = torch.sum(torch.log(1. - value * value + eps), dim=1)
        else:
            term = self._squash_correction(raw_value)
        out = out - term
        return out

    def _squash_correction(self, raw_value):
        ### Problem 2.B
        ### YOUR CODE HERE
        return torch.sum(np.log(4.) + 2. * raw_value - 2 * F.softplus(2. * raw_value), dim=1)
