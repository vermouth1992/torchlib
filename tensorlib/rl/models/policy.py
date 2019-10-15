import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions

from tensorlib.utils.distribution import TanhMultivariateNormalDiag


class BaseStochasticPolicyValue(tf.keras.Model):
    def __init__(self, shared=True, value_func=False, **kwargs):
        super(BaseStochasticPolicyValue, self).__init__()
        self.shared = shared
        self.action_model = self._create_feature_extractor()
        self.action_head = self._create_action_head()

        if value_func:
            if shared:
                self.value_model = self.action_model
            else:
                self.value_model = self._create_feature_extractor()
            self.value_head = tf.keras.layers.Dense(1)
        else:
            self.value_head = None

        # create tensorflow static graph for fast computation

        ob_shape = [None, self.state_dim]
        ac_shape = [None] if self.discrete else [None, self.action_dim]
        ac_dtype = tf.int64 if self.discrete else tf.float32

        @tf.function(input_signature=[tf.TensorSpec(shape=ob_shape)])
        def sample_action(state):
            print('Building sample action graph')
            return self(state)[0].sample()

        @tf.function(input_signature=[tf.TensorSpec(shape=ob_shape)])
        def select_action(state):
            print('Building select action graph')
            return self(state)[0].mean()

        @tf.function(input_signature=[tf.TensorSpec(shape=ob_shape), tf.TensorSpec(shape=ac_shape, dtype=ac_dtype)])
        def predict_log_prob(state, action):
            print('Building predict log probability graph')
            return self(state)[0].log_prob(action)

        @tf.function(input_signature=[tf.TensorSpec(shape=ob_shape)])
        def predict_action_log_prob(state):
            action_distribution = self(state)[0]
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)
            return action, log_prob

        # @tf.function(input_signature=[tf.TensorSpec(shape=ob_shape)])
        def predict_action_distribution(state):
            return self(state)[0]

        if value_func:
            @tf.function(input_signature=[tf.TensorSpec(shape=ob_shape)])
            def predict_value(state):
                print('Building predict value function graph')
                return self(state)[1]
        else:
            predict_value = None

        self.select_action = select_action
        self.predict_log_prob = predict_log_prob
        self.predict_value = predict_value
        self.predict_action_log_prob = predict_action_log_prob
        self.predict_action_distribution = predict_action_distribution

    def _create_feature_extractor(self):
        raise NotImplementedError

    def _create_action_head(self):
        raise NotImplementedError

    def call(self, state, training=None, mask=None):
        x = self.action_model(state)  # shape (T, feature_size)
        action_distribution = self.action_head(x)
        if self.value_head is not None:
            if self.shared:
                value = self.value_head(x)
            else:
                value = self.value_head(self.value_model(state))
            return action_distribution, tf.squeeze(value, axis=-1)
        else:
            return action_distribution, None


"""
Various Action Head
"""


class _NormalActionHead(tf.keras.layers.Layer):
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
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda a: ds.MultivariateNormalDiag(mu, tf.exp(logstd), allow_nan_stats=False),
        )(inputs=(mu, logstd))


class _TanhNormalActionHead(_NormalActionHead):
    def call(self, feature, training=None, mask=None):
        mu = self.mu_header(feature)
        logstd = self.log_std_header(feature)
        logstd = tf.clip_by_value(logstd, clip_value_min=self.log_std_range[0],
                                  clip_value_max=self.log_std_range[1])
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda a: TanhMultivariateNormalDiag(a[0], tf.exp(a[1])),
        )(inputs=(mu, logstd))


class _BetaActionHead(tf.keras.layers.Layer):
    def __init__(self, action_dim, log_std_range=(-20., 4.)):
        super(_BetaActionHead, self).__init__()
        self.log_std_range = log_std_range
        self.log_alpha_header = tf.keras.layers.Dense(action_dim)
        self.log_beta_header = tf.keras.layers.Dense(action_dim)

    def call(self, inputs, training=None, mask=None):
        log_alpha = self.log_alpha_header(inputs)
        log_beta = self.log_beta_header(inputs)
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda a: ds.Independent(
                distribution=ds.Beta(tf.exp(a[0]), tf.exp(a[1]), allow_nan_stats=False),
                reinterpreted_batch_ndims=1
            ))(inputs=(log_alpha, log_beta))


class _CategoricalActionHead(tf.keras.layers.Layer):
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
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda probs: ds.Categorical(probs=probs, allow_nan_stats=False, dtype=tf.int64)
        )(inputs=probs)


"""
Simple Policy for low dimensional state and action
"""


class _NormalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, shared_std, **kwargs):
        self.action_dim = action_dim
        self.shared_std = shared_std
        self.discrete = False
        super(_NormalPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _NormalActionHead(self.action_dim, self.shared_std)
        return action_header


class _TanhNormalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, shared_std, **kwargs):
        self.action_dim = action_dim
        self.shared_std = shared_std
        self.discrete = False
        super(_TanhNormalPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _TanhNormalActionHead(self.action_dim, self.shared_std)
        return action_header


class _BetaPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        self.discrete = False
        super(_BetaPolicy, self).__init__(**kwargs)

    def _create_action_head(self):
        action_header = _BetaActionHead(self.action_dim)
        return action_header


class _CategoricalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        self.discrete = True
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
                tf.keras.layers.Dense(self.nn_size),
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
                tf.keras.layers.Input(shape=(84, 84, self.num_channel)),
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
        self.build(input_shape=(None, state_dim))


class NormalNNPolicyValue(_NNPolicy, _NormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared, shared_std=False):
        super(NormalNNPolicyValue, self).__init__(shared=shared, value_func=True, shared_std=shared_std,
                                                  nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)
        self.build(input_shape=(None, state_dim))


class TanhNormalNNPolicy(_NNPolicy, _TanhNormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared_std=False):
        super(TanhNormalNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                 shared=False, value_func=False, shared_std=shared_std)
        self.build(input_shape=(None, state_dim))


class TanhNormalNNPolicyValue(_NNPolicy, _TanhNormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared, shared_std=False):
        super(TanhNormalNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                      shared=shared, value_func=True, shared_std=shared_std)
        self.build(input_shape=(None, state_dim))


class BetaNNPolicy(_NNPolicy, _BetaPolicy):
    def __init__(self, nn_size, state_dim, action_dim):
        super(BetaNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                           shared=False, value_func=False)
        self.build(input_shape=(None, state_dim))


class BetaNNPolicyValue(_NNPolicy, _BetaPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(BetaNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                shared=shared, value_func=True)
        self.build(input_shape=(None, state_dim))


class CategoricalNNPolicy(_NNPolicy, _CategoricalPolicy):
    def __init__(self, nn_size, state_dim, action_dim):
        super(CategoricalNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                  shared=False, value_func=False)


class CategoricalNNPolicyValue(_NNPolicy, _CategoricalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(CategoricalNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                       shared=shared, value_func=True)
        self.build(input_shape=(None, state_dim))


class AtariPolicy(_AtariCNNPolicy, _CategoricalPolicy):
    def __init__(self, num_channel, action_dim):
        super(AtariPolicy, self).__init__(num_channel=num_channel, action_dim=action_dim,
                                          shared=False, value_func=False)
        self.build(input_shape=(None, 84, 84, num_channel))

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
        self.build(input_shape=(None, 84, 84, num_channel))

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
