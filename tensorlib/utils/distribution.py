import tensorflow as tf
import tensorflow_probability as tfp
from tensorlib.utils.math import eps

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

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        result = super(TanhMultivariateNormalDiag, self).sample(sample_shape=sample_shape,
                                                                seed=seed,
                                                                name=name,
                                                                **kwargs)
        result = tf.clip_by_value(result, clip_value_min=-1 + eps, clip_value_max=1 - eps)
        return result
