import tensorflow as tf

"""
Low dimensional classic control module
"""


class BaseQModule(tf.keras.Model):
    def __init__(self, state_dim):
        super(BaseQModule, self).__init__()

        if isinstance(state_dim, int):
            state_dim = [state_dim]

        self.state_tensor_spec = tf.TensorSpec(shape=[None] + list(state_dim), dtype=tf.float32)
        self.action_tensor_spec = tf.TensorSpec(shape=[None], dtype=tf.int64)

        self.predict_value = tf.function(func=self.predict_value, input_signature=[self.state_tensor_spec])
        self.predict_value_with_action = tf.function(func=self.predict_value_with_action,
                                                     input_signature=[self.state_tensor_spec,
                                                                      self.action_tensor_spec])

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def predict_value(self, state):
        return self(state)

    def predict_value_with_action(self, state, action):
        q_values = self(state)  # (batch_size, action_dim)
        action = tf.one_hot(action, depth=self.action_dim, dtype=tf.float32)
        return tf.reduce_sum(tf.multiply(q_values, action), axis=-1)


class BaseCriticModule(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(BaseCriticModule, self).__init__()
        if isinstance(state_dim, int):
            state_dim = [state_dim]

        self.state_tensor_spec = tf.TensorSpec(shape=[None] + list(state_dim), dtype=tf.float32)
        self.action_tensor_spec = tf.TensorSpec(shape=[None, action_dim], dtype=tf.float32)

        self.predict_value_with_action = tf.function(func=self.predict_value_with_action,
                                                     input_signature=[self.state_tensor_spec,
                                                                      self.action_tensor_spec])

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def predict_value_with_action(self, state, action):
        return self((state, action))


class QModule(BaseQModule):
    def __init__(self, state_dim, action_dim, nn_size):
        super(QModule, self).__init__(state_dim=state_dim)
        self.model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(nn_size, activation='relu'),
                tf.keras.layers.Dense(nn_size, activation='relu'),
                tf.keras.layers.Dense(action_dim)
            ]
        )
        self.action_dim = action_dim

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class DuelQModule(BaseQModule):
    def __init__(self, state_dim, action_dim, nn_size):
        super(DuelQModule, self).__init__(state_dim=state_dim)
        self.fc1 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.adv_fc = tf.keras.layers.Dense(action_dim)
        self.value_fc = tf.keras.layers.Dense(1)
        self.action_dim = action_dim

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        adv = adv - tf.reduce_mean(adv, axis=-1, keepdim=True)
        x = value + adv
        return x


class CriticModule(BaseCriticModule):
    def __init__(self, nn_size, state_dim, action_dim):
        super(CriticModule, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self.fc1 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

    def call(self, inputs, training=None, mask=None):
        x = tf.concat(inputs, axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = tf.squeeze(x, axis=-1)
        return x


class BaseDoubleCriticModule(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BaseDoubleCriticModule, self).__init__()
        self.critic1 = self._create_model(**kwargs)
        self.critic2 = self._create_model(**kwargs)

        self.state_tensor_spec = self.critic1.state_tensor_spec
        self.action_tensor_spec = self.critic1.action_tensor_spec

        self.predict_value_with_action = self.__make_state_action_tf(self.predict_value_with_action)
        self.predict_min_value_with_action = self.__make_state_action_tf(self.predict_min_value_with_action)

    def call(self, inputs, training=None, mask=None):
        return self.critic1(inputs), self.critic2(inputs)

    def _create_model(self, **kwargs):
        raise NotImplementedError

    @property
    def __make_state_action_tf(self):
        return lambda func: tf.function(func=func, input_signature=[self.state_tensor_spec,
                                                                    self.action_tensor_spec])

    def predict_value_with_action(self, state, action):
        return self.critic1.predict_value_with_action(state, action), \
               self.critic2.predict_value_with_action(state, action)

    def predict_min_value_with_action(self, state, action):
        value1, value2 = self.predict_value_with_action(state, action)
        return tf.minimum(value1, value2)


class BaseDoubleQModule(BaseDoubleCriticModule):
    def __init__(self, **kwargs):
        super(BaseDoubleQModule, self).__init__(**kwargs)
        self.predict_value = self.__make_state_tf(self.predict_value)
        self.predict_min_value = self.__make_state_tf(self.predict_min_value)

    @property
    def __make_state_tf(self):
        return lambda func: tf.function(func=func, input_signature=[self.state_tensor_spec])

    def predict_value(self, state):
        return self.critic1.predict_value(state), self.critic2.predict_value(state)

    def predict_min_value(self, state):
        value1, value2 = self.predict_value(state)
        return tf.minimum(value1, value2)


class DoubleCriticModule(BaseDoubleCriticModule):
    def __init__(self, nn_size, state_dim, action_dim):
        super(DoubleCriticModule, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)
        self.build(input_shape=[(None, state_dim), (None, action_dim)])

    def _create_model(self, nn_size, state_dim, action_dim):
        return CriticModule(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class DoubleQModule(BaseDoubleQModule):
    def __init__(self, nn_size, state_dim, action_dim):
        super(DoubleQModule, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)
        self.build(input_shape=(None, state_dim))

    def _create_model(self, nn_size, state_dim, action_dim):
        return QModule(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class ValueModule(tf.keras.Model):
    def __init__(self, nn_size, state_dim):
        super(ValueModule, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        self.state_tensor_spec = tf.TensorSpec(shape=[None, state_dim], dtype=tf.float32)

        self.predict_value = self.__make_state_tf(self.predict_value)

    @property
    def __make_state_tf(self):
        return lambda func: tf.function(func=func, input_signature=[self.state_tensor_spec])

    def predict_value(self, state):
        return self(state)

    def call(self, state, training=None, mask=None):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = tf.squeeze(x, axis=-1)
        return x


"""
Atari Policy
"""


class AtariQModule(BaseQModule):
    def __init__(self, frame_history_len, action_dim):
        super(AtariQModule, self).__init__(state_dim=(64, 64, frame_history_len))
        self.model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(action_dim)
            ]
        )

    def call(self, state, training=None, mask=None):
        state = state / 255.0
        return self.model(state)


class DoubleAtariQModule(BaseDoubleQModule):
    def __init__(self, frame_history_len, action_dim):
        super(DoubleAtariQModule, self).__init__(frame_history_len=frame_history_len, action_dim=action_dim)

    def _create_model(self, frame_history_len, action_dim):
        return AtariQModule(frame_history_len=frame_history_len, action_dim=action_dim)


class AtariDuelQModule(BaseQModule):
    def __init__(self, frame_history_len, action_dim):
        super(AtariDuelQModule, self).__init__(state_dim=(84, 84, frame_history_len))
        self.model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(action_dim)
            ]
        )
        self.adv_fc = tf.keras.layers.Dense(action_dim)
        self.value_fc = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        inputs = inputs / 255.0
        x = self.model(inputs)
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        adv = adv - tf.reduce_mean(adv, axis=-1, keepdim=True)
        x = value + adv
        return x


class DoubleAtariDuelQModule(BaseDoubleQModule):
    def __init__(self, frame_history_len, action_dim):
        super(DoubleAtariDuelQModule, self).__init__(frame_history_len=frame_history_len, action_dim=action_dim)

    def _create_model(self, frame_history_len, action_dim):
        return AtariDuelQModule(frame_history_len=frame_history_len, action_dim=action_dim)
