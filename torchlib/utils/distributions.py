"""
Rewrite Pytorch builtin distribution function to favor policy gradient
1. Normal distribution with multiple mean and std as a single distribution
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Transform, constraints, TransformedDistribution

from torchlib.common import eps


class TanhTransform(Transform):
    domain = constraints.real
    codomain = constraints.interval(-1., 1.)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        return 0.5 * (torch.log1p(y + eps) - torch.log1p(-y + eps))

    def log_abs_det_jacobian(self, x, y):
        return np.log(4.) + 2. * x - 2 * F.softplus(2. * x)


class TanhNormal(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        super(TanhNormal, self).__init__(base_dist, TanhTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhNormal, _instance)
        return super(TanhNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale


def create_independent_normal(loc, scale, validate_args=None):
    return Independent(Normal(loc=loc, scale=scale, validate_args=validate_args), len(loc.shape) - 1,
                       validate_args=validate_args)


def create_independent_tanh_normal(loc, scale, validate_args=None):
    return Independent(TanhNormal(loc=loc, scale=scale, validate_args=validate_args), len(loc.shape) - 1,
                       validate_args=validate_args)
