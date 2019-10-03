import tensorflow_probability as tfp

ds = tfp.distributions


class TanhMultivariateNormalDiag(ds.TransformedDistribution):
    def __init__(self, loc, scale_diag):
        super(TanhMultivariateNormalDiag, self).__init__(
            distribution=ds.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag, allow_nan_stats=False),
            bijector=tfp.bijectors.Tanh()
        )

    def entropy(self, name='entropy', **kwargs):
        """ Use negative log probability to estimate the entropy

        Args:
            name:
            **kwargs:

        Returns:

        """
        return -self.log_prob(self.sample())
