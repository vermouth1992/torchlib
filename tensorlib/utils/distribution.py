import tensorflow_probability as tfp
import tensorflow as tf

ds = tfp.distributions


def get_tanh_multivariate_normal_diag(loc, scale_diag):
    return ds.TransformedDistribution(
        distribution=ds.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag),
        bijector=tfp.bijectors.Tanh()
    )