"""
Common architecture for policy gradient.
The architecture follows the same rule:
1. A feature extractor layer. NN for low dimensional state space. CNN for image based state.
2. An optional GRU for recurrent policy.
3. A action header and a value header.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchlib.common import FloatTensor
from torchlib.deep_rl.utils.distributions import FixedNormal
from torchlib.utils.torch_layer_utils import conv2d_bn_relu_block, linear_bn_relu_block, Flatten


class BasePolicy(nn.Module):
    def __init__(self, recurrent, hidden_size):
        super(BasePolicy, self).__init__()
        self.recurrent = recurrent
        self.model = self._create_feature_extractor()
        feature_output_size = self._calculate_feature_output_size()
        if self.recurrent:
            self.gru = nn.GRU(feature_output_size, hidden_size, batch_first=False)

        if recurrent:
            feature_output_size = hidden_size

        self.value_head = nn.Linear(feature_output_size, 1)
        self.action_head = self._create_action_head(feature_output_size)

    def _calculate_feature_output_size(self):
        raise NotImplementedError

    def _create_feature_extractor(self):
        raise NotImplementedError

    def _create_action_head(self, feature_output_size):
        raise NotImplementedError

    def forward(self, state, hidden):
        """ This method can serve as two cases
        if state.shape[0] == hidden.shape[0], then we treat state and hidden as batch input with timestamp=1
        else we treat state as consecutive T timestamp and hidden as initial hidden state.

        Args:
            state: (T, dim_1, dim_2, ..., dim_n)
            hidden: (1, hidden_size)

        Returns: action, hidden, value

        """
        x = self.model.forward(state)  # shape (T, feature_size)
        if self.recurrent:
            if state.shape[0] == hidden.shape[0]:
                axis = 0  # expand on seq_length
            else:
                axis = 1  # expand on batch_size
            x, hidden = self.gru.forward(x.unsqueeze(axis), hidden.unsqueeze(axis))  # assume batch size is 1
            x = x.squeeze(axis)
            hidden = hidden.squeeze(1)
        action = self.action_head.forward(x)
        value = self.value_head.forward(x)
        return action, hidden, value.squeeze(-1)


"""
Simple Policy for low dimensional state and action
"""


class ContinuousActionHead(nn.Module):
    def __init__(self, feature_output_size, action_dim):
        super(ContinuousActionHead, self).__init__()
        random_number = np.random.randn()
        self.logstd = torch.nn.Parameter(torch.tensor(random_number, requires_grad=True).type(FloatTensor))

        self.action_head = nn.Sequential(
            nn.Linear(feature_output_size, action_dim),
            nn.Tanh()
        )

    def forward(self, feature):
        mu = self.action_head.forward(feature)
        return FixedNormal(mu, torch.exp(self.logstd))


class DiscreteActionHead(nn.Module):
    def __init__(self, feature_output_size, action_dim):
        super(DiscreteActionHead, self).__init__()
        self.action_head = nn.Sequential(
            nn.Linear(feature_output_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, feature):
        probs = self.action_head.forward(feature)
        return Categorical(probs=probs)


class ContinuousPolicy(BasePolicy):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(ContinuousPolicy, self).__init__(**kwargs)

    def _create_action_head(self, feature_output_size):
        action_header = ContinuousActionHead(feature_output_size, self.action_dim)
        return action_header


class DiscretePolicy(BasePolicy):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(DiscretePolicy, self).__init__(**kwargs)

    def _create_action_head(self, feature_output_size):
        action_header = DiscreteActionHead(feature_output_size, self.action_dim)
        return action_header


class NNPolicy(BasePolicy):
    def __init__(self, nn_size, state_dim, **kwargs):
        self.nn_size = nn_size
        self.state_dim = state_dim
        super(NNPolicy, self).__init__(**kwargs),

    def _calculate_feature_output_size(self):
        return self.nn_size

    def _create_feature_extractor(self):
        state_dim = self.state_dim
        nn_size = self.nn_size
        model = nn.Sequential(
            nn.Linear(state_dim, nn_size),
            nn.ReLU(),
            nn.Linear(nn_size, nn_size),
            nn.ReLU()
        )
        return model


class AtariCNNPolicy(BasePolicy):
    def __init__(self, num_channel, **kwargs):
        self.num_channel = num_channel
        super(AtariCNNPolicy, self).__init__(**kwargs)

    def _create_feature_extractor(self):
        feature = nn.Sequential(
            *conv2d_bn_relu_block(self.num_channel, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
        )
        return feature

    def _calculate_feature_output_size(self):
        return 512


class ContinuousNNPolicy(NNPolicy, ContinuousPolicy):
    def __init__(self, recurrent, hidden_size, nn_size, state_dim, action_dim):
        super(ContinuousNNPolicy, self).__init__(recurrent=recurrent, hidden_size=hidden_size,
                                                 nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class DiscreteNNPolicy(NNPolicy, DiscretePolicy):
    def __init__(self, recurrent, hidden_size, nn_size, state_dim, action_dim):
        super(DiscreteNNPolicy, self).__init__(recurrent=recurrent, hidden_size=hidden_size,
                                               nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class AtariPolicy(AtariCNNPolicy, DiscretePolicy):
    def __init__(self, recurrent, hidden_size, num_channel, action_dim):
        super(AtariPolicy, self).__init__(recurrent=recurrent, hidden_size=hidden_size, num_channel=num_channel,
                                          action_dim=action_dim)

    def forward(self, state, hidden):
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        return super(AtariPolicy, self).forward(state, hidden)
