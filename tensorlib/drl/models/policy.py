import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions

from tensorlib.utils.distribution import TanhMultivariateNormalDiag


class BaseStochasticPolicyValue(tf.keras.Model):
    def __init__(self, shared=True, value_func=False, **kwargs):
        super(BaseStochasticPolicyValue, self).__init__()
        self.shared = shared
        self.action_model = self._create_feature_extractor()
        if shared:
            self.value_model = self.action_model
        else:
            self.value_model = self._create_feature_extractor()
        self.action_head = self._create_action_head()

        if value_func:
            self.value_head = tf.keras.layers.Dense(1)
        else:
            self.value_head = None

    def _create_feature_extractor(self):
        raise NotImplementedError

    def _create_action_head(self):
        raise NotImplementedError

    def call(self, state, training=None, mask=None):
        x = self.action_model(state)  # shape (T, feature_size)
        action = self.action_head(x)
        if self.value_head is not None:
            if self.shared:
                value = self.value_head(x)
            else:
                value = self.value_head(self.value_model(state))
            return action, tf.squeeze(value, axis=-1)
        else:
            return action


"""
Various Action Head
"""


class _NormalActionHead(tf.keras.Model):
    def __init__(self, action_dim, shared_std=False, log_std_range=(-20., 2.)):
        super(_NormalActionHead, self).__init__()
        self.log_std_range = log_std_range
        initial_range = 3e-3
        self.mu_header = tf.keras.layers.Dense(action_dim,
                                               kernel_initializer=tf.random_uniform_initializer(-initial_range,
                                                                                                initial_range))
        if shared_std:
            self.log_std_header = tf.Variable(tf.random.uniform([action_dim],
                                                                minval=-initial_range,
                                                                maxval=initial_range))
        else:
            self.log_std_header = tf.keras.layers.Dense(action_dim,
                                                        kernel_initializer=tf.random_uniform_initializer(-initial_range,
                                                                                                         initial_range))

    def call(self, feature, training=None, mask=None):
        mu = self.mu_header(feature)
        logstd = self.log_std_header(feature)
        logstd = tf.clip_by_value(logstd, clip_value_min=self.log_std_range[0],
                                  clip_value_max=self.log_std_range[1])
        return ds.MultivariateNormalDiag(mu, tf.exp(logstd), allow_nan_stats=False)


class _TanhNormalActionHead(_NormalActionHead):
    def call(self, feature, training=None, mask=None):
        mu = self.mu_header(feature)
        logstd = self.log_std_header(feature)
        logstd = tf.clip_by_value(logstd, clip_value_min=self.log_std_range[0],
                                  clip_value_max=self.log_std_range[1])
        return TanhMultivariateNormalDiag(mu, tf.exp(logstd))


class _BetaActionHead(tf.keras.Model):
    def __init__(self, action_dim, log_std_range=(-20., 4.)):
        super(_BetaActionHead, self).__init__()
        self.log_std_range = log_std_range
        self.log_alpha_header = tf.keras.layers.Dense(action_dim)
        self.log_beta_header = tf.keras.layers.Dense(action_dim)

    def call(self, inputs, training=None, mask=None):
        log_alpha = self.log_alpha_header(inputs)
        log_beta = self.log_beta_header(inputs)
        return ds.Independent(
            distribution=ds.Beta(tf.exp(log_alpha), tf.exp(log_beta), allow_nan_stats=False),
            reinterpreted_batch_ndims=1
        )


class _CategoricalActionHead(tf.keras.Model):
    def __init__(self, action_dim):
        super(_CategoricalActionHead, self).__init__()
        self.action_head = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(action_dim),
                tf.keras.layers.Softmax(axis=-1)
            ]
        )

    def call(self, inputs, training=None, mask=None):
        probs = self.action_head(inputs)
        return ds.Categorical(probs=probs, allow_nan_stats=False)


"""
Simple Policy for low dimensional state and action
"""


class _NormalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, shared_std, **kwargs):
        self.action_dim = action_dim
        self.shared_std = shared_std
        super(_NormalPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _NormalActionHead(self.action_dim, self.shared_std)
        return action_header


class _TanhNormalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, shared_std, **kwargs):
        self.action_dim = action_dim
        self.shared_std = shared_std
        super(_TanhNormalPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _TanhNormalActionHead(self.action_dim, self.shared_std)
        return action_header


class _BetaPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(_BetaPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _BetaActionHead(self.action_dim)
        return action_header


class _CategoricalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(_CategoricalPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _CategoricalActionHead(self.action_dim)
        return action_header


class _NNPolicy(BaseStochasticPolicyValue):
    def __init__(self, nn_size, state_dim, **kwargs):
        self.nn_size = nn_size
        self.state_dim = state_dim
        super(_NNPolicy, self).__init__(**kwargs),

    def _create_feature_extractor(self):
        model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(self.nn_size, input_shape=(self.state_dim,)),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.nn_size),
                tf.keras.layers.ReLU()
            ]
        )
        return model


class _AtariCNNPolicy(BaseStochasticPolicyValue):
    def __init__(self, num_channel, **kwargs):
        self.num_channel = num_channel
        super(_AtariCNNPolicy, self).__init__(**kwargs)

    def _create_feature_extractor(self):
        feature = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Input(shape=(64, 64, self.num_channel)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu')
            ]
        )
        return feature


class NormalNNPolicy(_NNPolicy, _NormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared, shared_std=False):
        super(NormalNNPolicy, self).__init__(shared=shared, value_func=False, shared_std=shared_std,
                                             nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class NormalNNPolicyValue(_NNPolicy, _NormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared, shared_std=False):
        super(NormalNNPolicyValue, self).__init__(shared=shared, value_func=True, shared_std=shared_std,
                                                  nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class TanhNormalNNPolicy(_NNPolicy, _TanhNormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared_std=False):
        super(TanhNormalNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                 shared=False, value_func=False, shared_std=shared_std)


class TanhNormalNNPolicyValue(_NNPolicy, _TanhNormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared, shared_std=False):
        super(TanhNormalNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                      shared=shared, value_func=True, shared_std=shared_std)


class BetaNNPolicy(_NNPolicy, _BetaPolicy):
    def __init__(self, nn_size, state_dim, action_dim):
        super(BetaNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                           shared=False, value_func=False)


class BetaNNPolicyValue(_NNPolicy, _BetaPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(BetaNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                shared=shared, value_func=True)


class CategoricalNNPolicy(_NNPolicy, _CategoricalPolicy):
    def __init__(self, nn_size, state_dim, action_dim):
        super(CategoricalNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                  shared=False, value_func=False)


class CategoricalNNPolicyValue(_NNPolicy, _CategoricalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(CategoricalNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                       shared=shared, value_func=True)


class AtariPolicy(_AtariCNNPolicy, _CategoricalPolicy):
    def __init__(self, num_channel, action_dim):
        super(AtariPolicy, self).__init__(num_channel=num_channel, action_dim=action_dim,
                                          shared=False, value_func=False)

    def _normalize_obs(self, state):
        state = state / 255.0
        return state

    def call(self, state, training=None, mask=None):
        state = self._normalize_obs(state)
        super(AtariPolicy, self).call(state, training, mask)


class AtariPolicyValue(_AtariCNNPolicy, _CategoricalPolicy):
    def __init__(self, num_channel, action_dim):
        super(AtariPolicyValue, self).__init__(num_channel=num_channel, action_dim=action_dim,
                                               shared=True, value_func=True)

    def _normalize_obs(self, state):
        state = state / 255.0
        return state

    def call(self, state, training=None, mask=None):
        state = self._normalize_obs(state)
        super(AtariPolicyValue, self).call(state, training, mask)


"""
Deterministic Policy
"""


def create_deterministic_actor(nn_size, state_dim, action_dim, output_activation=tf.tanh):
    return tf.keras.Sequential(
        layers=[
            tf.keras.layers.Dense(nn_size, input_shape=(state_dim,), activation='relu'),
            tf.keras.layers.Dense(nn_size, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation=output_activation,
                                  kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        ]
    )
