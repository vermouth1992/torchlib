"""
Rewrite Pytorch builtin distribution function to favor policy gradient
1. Normal distribution with multiple mean and std as a single distribution
"""

from torch.distributions import Normal


class FixedNormal(Normal):
    def log_prob(self, value):
        result = super(FixedNormal, self).log_prob(value)
        return result.sum(-1)

    def entropy(self):
        return super(FixedNormal, self).entropy().sum(-1)
