import tensorflow as tf

"""
Low dimensional classic control module
"""


class QModule(tf.keras.Model):
    def __init__(self, nn_size, state_dim, action_dim):
        super(QModule, self).__init__()
        self.model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(nn_size, activation='relu'),
                tf.keras.layers.Dense(nn_size, activation='relu'),
                tf.keras.layers.Dense(action_dim)
            ]
        )

        self.predict_value = tf.function(func=self.predict_value,
                                         input_signature=[tf.TensorSpec(shape=[None, state_dim])])
        self.predict_value_with_action = tf.function(func=self.predict_value_with_action,
                                                     input_signature=[tf.TensorSpec(shape=[None, state_dim]),
                                                                      tf.TensorSpec(shape=[None, action_dim])])

    def call(self, state, training=None, mask=None):
        return self.model(state)

    def predict_value(self, state):
        return self(state)

    def predict_value_with_action(self, state, action):
        q_values = self(state)
        return tf.gather(q_values, tf.expand_dims(action, axis=1), batch_dims=1)


class DuelQModule(tf.keras.Model):
    def __init__(self, nn_size, state_dim, action_dim):
        super(DuelQModule, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.adv_fc = tf.keras.layers.Dense(action_dim)
        self.value_fc = tf.keras.layers.Dense(1)

        self.predict_value = tf.function(func=self.predict_value,
                                         input_signature=[tf.TensorSpec(shape=[None, state_dim])])
        self.predict_value_with_action = tf.function(func=self.predict_value_with_action,
                                                     input_signature=[tf.TensorSpec(shape=[None, state_dim]),
                                                                      tf.TensorSpec(shape=[None, action_dim])])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        adv = adv - tf.reduce_mean(adv, axis=-1, keepdim=True)
        x = value + adv
        return x

    def predict_value(self, state):
        return self(state)

    def predict_value_with_action(self, state, action):
        return tf.gather(self.predict_value(state), tf.expand_dims(action, axis=1), batch_dims=1)


class CriticModule(tf.keras.Model):
    def __init__(self, nn_size, state_dim, action_dim):
        super(CriticModule, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(nn_size, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_dim,
                                         kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        self.predict_value = tf.function(func=self.predict_value,
                                         input_signature=[tf.TensorSpec(shape=[None, state_dim]),
                                                          tf.TensorSpec(shape=[None, action_dim])])

    def call(self, inputs, training=None, mask=None):
        x = tf.concat(inputs, axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = tf.squeeze(x, axis=-1)
        return x

    def predict_value(self, state, action):
        return self((state, action))


class DoubleCriticModule(tf.keras.Model):
    def __init__(self, nn_size, state_dim, action_dim):
        super(DoubleCriticModule, self).__init__()
        self.critic1 = CriticModule(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)
        self.critic2 = CriticModule(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)

        self.predict_value = tf.function(func=self.predict_value,
                                         input_signature=[tf.TensorSpec(shape=[None, state_dim]),
                                                          tf.TensorSpec(shape=[None, action_dim])])

        self.predict_min_value = tf.function(func=self.predict_min_value,
                                             input_signature=[tf.TensorSpec(shape=[None, state_dim]),
                                                              tf.TensorSpec(shape=[None, action_dim])])

    def predict_value(self, state, action):
        return self.critic1.predict_value(state, action), self.critic2.predict_value(state, action)

    def predict_min_value(self, state, action):
        value1, value2 = self.predict_value(state, action)
        return tf.minimum(value1, value2)


class DoubleQModule(tf.keras.Model):
    def __init__(self, nn_size, state_dim, action_dim):
        super(DoubleQModule, self).__init__()
        self.critic1 = QModule(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)
        self.critic2 = QModule(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)

    def make_state_tf_function(self, func):
        return tf.function(func=func, input_signature=[tf.TensorSpec(shape=[None, self.state_dim])])

    @make_state_tf_function
    def predict_value(self, state):
        return self.critic1.predict_value(state), self.critic2.predict_value(state)

    def predict_min_value(self, state):
        value1, value2 = self.predict_value(state)
        return tf.minimum(value1, value2)

    def predict_value_with_action(self, state, action):
        return self.critic1.predict_value_with_action(state, action), \
               self.critic2.predict_value_with_action(state, action)

    def predict_min_value_with_action(self, state, action):
        value1, value2 = self.predict_value_with_action(state, action)
        return tf.minimum(value1, value2)


class ValueModule(nn.Module):
    def __init__(self, size, state_dim):
        super(ValueModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(size, size)
        fanin_init(self.fc2)
        self.fc3 = nn.Linear(size, 1)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.squeeze(x, dim=-1)
        return x


"""
Atari Policy
"""


class AtariQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(AtariQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
            nn.Linear(512, action_dim)
        )

    def forward(self, state, action=None):
        state = state.type(FloatTensor)
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        out = self.model.forward(state)
        if action is not None:
            out = out.gather(1, action.unsqueeze(1)).squeeze()
        return out


class DoubleAtariQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(DoubleAtariQModule, self).__init__()
        self.critic1 = AtariQModule(frame_history_len, action_dim)
        self.critic2 = AtariQModule(frame_history_len, action_dim)

    def forward(self, state, action=None, minimum=True):
        x1 = self.critic1.forward(state, action)
        x2 = self.critic2.forward(state, action)
        if minimum:
            return torch.min(x1, x2)
        return x1, x2


class AtariDuelQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(AtariDuelQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
        )
        self.adv_fc = nn.Linear(512, action_dim)
        self.value_fc = nn.Linear(512, 1)

    def forward(self, state, action=None):
        state = state.type(FloatTensor)
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        state = self.model.forward(state)
        value = self.value_fc(state)
        adv = self.adv_fc(state)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        out = value + adv
        if action is not None:
            out = out.gather(1, action.unsqueeze(1)).squeeze()
        return out


class DoubleAtariDuelQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(DoubleAtariDuelQModule, self).__init__()
        self.critic1 = AtariDuelQModule(frame_history_len, action_dim)
        self.critic2 = AtariDuelQModule(frame_history_len, action_dim)

    def forward(self, state, action=None, minimum=True):
        x1 = self.critic1.forward(state, action)
        x2 = self.critic2.forward(state, action)
        if minimum:
            return torch.min(x1, x2)
        return x1, x2
