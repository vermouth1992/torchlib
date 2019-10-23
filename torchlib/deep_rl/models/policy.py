"""
Common architecture for policy gradient.
The architecture follows the same rule:
1. A feature extractor layer. NN for low dimensional state space. CNN for image based state.
2. An optional GRU for recurrent policy.
3. A action header and a value header.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torchlib.common import FloatTensor
from torchlib.utils.distributions import IndependentTanhNormal, IndependentNormal, IndependentRescaledBeta
from torchlib.utils.layers import conv2d_bn_relu_block, linear_bn_relu_block, Flatten, Squeeze


class BaseStochasticPolicyValue(nn.Module):
    def __init__(self, shared=True, value_func=False, **kwargs):
        super(BaseStochasticPolicyValue, self).__init__()
        self.shared = shared
        self.action_model = self._create_feature_extractor()
        if shared:
            self.value_model = self.action_model
        else:
            self.value_model = self._create_feature_extractor()
        feature_output_size = self._calculate_feature_output_size()
        self.action_head = self._create_action_head(feature_output_size)

        if value_func:
            self.value_head = nn.Sequential(
                nn.Linear(feature_output_size, 1),
                Squeeze(dim=-1),
            )
        else:
            self.value_head = None

    def _calculate_feature_output_size(self):
        raise NotImplementedError

    def _create_feature_extractor(self):
        raise NotImplementedError

    def _create_action_head(self, feature_output_size):
        raise NotImplementedError

    def forward_action(self, state):
        x = self.action_model.forward(state)  # shape (T, feature_size)
        action = self.action_head.forward(x)
        return action

    def forward_value(self, state):
        if self.value_head is None:
            raise NotImplementedError
        x = self.value_model.forward(state)  # shape (T, feature_size)
        value = self.value_head.forward(x)
        return value

    def forward(self, state):
        """ This method can serve as two cases
        if state.shape[0] == hidden.shape[0], then we treat state and hidden as batch input with timestamp=1
        else we treat state as consecutive T timestamp and hidden as initial hidden state.

        Args:
            state: (T, dim_1, dim_2, ..., dim_n)
            hidden: (1, hidden_size)

        Returns: action, hidden, value

        """
        x = self.action_model.forward(state)  # shape (T, feature_size)
        action = self.action_head.forward(x)
        if self.value_head is not None:
            if self.shared:
                value = self.value_head.forward(x)
            else:
                value = self.forward_value(state)
            return action, value
        else:
            return action


"""
Various Action Head
"""


class _NormalActionHead(nn.Module):
    def __init__(self, feature_output_size, action_dim, log_std_range=(-20., 2.)):
        super(_NormalActionHead, self).__init__()
        self.log_std_range = log_std_range
        self.mu_header = nn.Linear(feature_output_size, action_dim)
        self.log_std_header = nn.Linear(feature_output_size, action_dim)
        torch.nn.init.uniform_(self.mu_header.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.log_std_header.weight.data, -3e-3, 3e-3)

    def forward(self, feature):
        mu = self.mu_header.forward(feature)
        logstd = self.log_std_header.forward(feature)
        logstd = torch.clamp(logstd, min=self.log_std_range[0], max=self.log_std_range[1])
        return IndependentNormal(mu, torch.exp(logstd))


class _TanhNormalActionHead(nn.Module):
    def __init__(self, feature_output_size, action_dim, log_std_range=(-20., 2.)):
        super(_TanhNormalActionHead, self).__init__()
        self.log_std_range = log_std_range
        self.mu_header = nn.Linear(feature_output_size, action_dim)
        self.log_std_header = nn.Linear(feature_output_size, action_dim)
        torch.nn.init.uniform_(self.mu_header.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.log_std_header.weight.data, -3e-3, 3e-3)

    def forward(self, feature):
        mu = self.mu_header.forward(feature)
        logstd = self.log_std_header.forward(feature)
        logstd = torch.clamp(logstd, min=self.log_std_range[0], max=self.log_std_range[1])
        return IndependentTanhNormal(mu, torch.exp(logstd))


class _BetaActionHead(nn.Module):
    def __init__(self, feature_output_size, action_dim, log_std_range=(-20., 4.)):
        super(_BetaActionHead, self).__init__()
        self.log_std_range = log_std_range
        self.log_alpha_header = nn.Linear(feature_output_size, action_dim)
        self.log_beta_header = nn.Linear(feature_output_size, action_dim)
        torch.nn.init.uniform_(self.log_alpha_header.weight.data, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.log_beta_header.weight.data, -3e-3, 3e-3)

    def forward(self, feature):
        log_alpha = self.log_alpha_header.forward(feature)
        log_beta = self.log_beta_header.forward(feature)
        # log_alpha = torch.clamp(log_alpha, min=self.log_std_range[0], max=self.log_std_range[1])
        # log_beta = torch.clamp(log_beta, min=self.log_std_range[0], max=self.log_std_range[1])
        return IndependentRescaledBeta(torch.exp(log_alpha), torch.exp(log_beta))


class _CategoricalActionHead(nn.Module):
    def __init__(self, feature_output_size, action_dim):
        super(_CategoricalActionHead, self).__init__()
        self.action_head = nn.Sequential(
            nn.Linear(feature_output_size, action_dim),
        )

    def forward(self, feature):
        logits = self.action_head.forward(feature)
        return Categorical(logits=logits)


"""
Simple Policy for low dimensional state and action
"""


class _NormalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(_NormalPolicy, self).__init__(**kwargs)

    def _create_action_head(self, feature_output_size):
        action_header = _NormalActionHead(feature_output_size, self.action_dim)
        return action_header


class _TanhNormalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(_TanhNormalPolicy, self).__init__(**kwargs)

    def _create_action_head(self, feature_output_size):
        action_header = _TanhNormalActionHead(feature_output_size, self.action_dim)
        return action_header


class _BetaPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(_BetaPolicy, self).__init__(**kwargs)

    def _create_action_head(self, feature_output_size):
        action_header = _BetaActionHead(feature_output_size, self.action_dim)
        return action_header


class _CategoricalPolicy(BaseStochasticPolicyValue):
    def __init__(self, action_dim, **kwargs):
        self.action_dim = action_dim
        super(_CategoricalPolicy, self).__init__(**kwargs)

    def _create_action_head(self, feature_output_size):
        action_header = _CategoricalActionHead(feature_output_size, self.action_dim)
        return action_header


class _NNPolicy(BaseStochasticPolicyValue):
    def __init__(self, nn_size, state_dim, **kwargs):
        self.nn_size = nn_size
        self.state_dim = state_dim
        super(_NNPolicy, self).__init__(**kwargs),

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


class _AtariCNNPolicy(BaseStochasticPolicyValue):
    def __init__(self, num_channel, **kwargs):
        self.num_channel = num_channel
        super(_AtariCNNPolicy, self).__init__(**kwargs)

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


class NormalNNPolicy(_NNPolicy, _NormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(NormalNNPolicy, self).__init__(shared=shared, value_func=False,
                                             nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class NormalNNPolicyValue(_NNPolicy, _NormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(NormalNNPolicyValue, self).__init__(shared=shared, value_func=True,
                                                  nn_size=nn_size, state_dim=state_dim, action_dim=action_dim)


class TanhNormalNNPolicy(_NNPolicy, _TanhNormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim):
        super(TanhNormalNNPolicy, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                 shared=False, value_func=False)


class TanhNormalNNPolicyValue(_NNPolicy, _TanhNormalPolicy):
    def __init__(self, nn_size, state_dim, action_dim, shared):
        super(TanhNormalNNPolicyValue, self).__init__(nn_size=nn_size, state_dim=state_dim, action_dim=action_dim,
                                                      shared=shared, value_func=True)


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
        state = state.type(FloatTensor)
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        return state

    def forward(self, state):
        state = self._normalize_obs(state)
        return super(AtariPolicy, self).forward(state)

    def forward_action(self, state):
        state = self._normalize_obs(state)
        return super(AtariPolicy, self).forward_action(state)

    def forward_value(self, state):
        raise NotImplementedError


class AtariPolicyValue(_AtariCNNPolicy, _CategoricalPolicy):
    def __init__(self, num_channel, action_dim):
        super(AtariPolicyValue, self).__init__(num_channel=num_channel, action_dim=action_dim,
                                               shared=True, value_func=True)

    def _normalize_obs(self, state):
        state = state.type(FloatTensor)
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        return state

    def forward(self, state):
        state = self._normalize_obs(state)
        return super(AtariPolicyValue, self).forward(state)

    def forward_action(self, state):
        state = self._normalize_obs(state)
        return super(AtariPolicyValue, self).forward_action(state)

    def forward_value(self, state):
        state = self._normalize_obs(state)
        return super(AtariPolicyValue, self).forward_value(state)


"""
Deterministic Policy
"""
from torchlib.utils.weight import fanin_init


class ActorModule(nn.Module):
    """
    Actor module for various algorithms including DDPG and Imitation Learning
    """

    def __init__(self, size, state_dim, action_dim, output_activation=torch.tanh):
        super(ActorModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(size, size)
        fanin_init(self.fc2)
        self.fc3 = nn.Linear(size, action_dim)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
