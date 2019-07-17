"""
Define model based wrapper for other well written environments.
Use to define cost_fn for state transition (state, action, next_state)
without learning reward functions.
"""

import gym
import numpy as np
import torch

from torchlib.common import LongTensor, FloatTensor


class ModelBasedWrapper(gym.Wrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        raise NotImplementedError

    def cost_fn_torch_batch(self, states, actions, next_states):
        raise NotImplementedError

    def cost_fn(self, state, action, next_state):
        if isinstance(state, torch.Tensor):
            states = torch.unsqueeze(state, dim=0)
            actions = torch.unsqueeze(action, dim=0)
            next_states = torch.unsqueeze(next_state, dim=0)
            return self.cost_fn_torch_batch(states, actions, next_states)[0]
        elif isinstance(state, np.ndarray):
            states = np.expand_dims(state, axis=0)
            actions = np.expand_dims(action, axis=0)
            next_states = np.expand_dims(next_state, axis=0)
            return self.cost_fn_numpy_batch(states, actions, next_states)[0]

        else:
            raise ValueError('Unknown data type {}'.format(type(state)))

    def cost_fn_batch(self, states, actions, next_states):
        if isinstance(states, torch.Tensor):
            return self.cost_fn_torch_batch(states, actions, next_states)

        elif isinstance(states, np.ndarray):
            return self.cost_fn_numpy_batch(states, actions, next_states)

        else:
            raise ValueError('Unknown data type {}'.format(type(states)))


class ModelBasedCartPoleWrapper(ModelBasedWrapper):
    version = 'v2'

    def cost_fn_numpy_batch_v1(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        x_done = np.logical_or(x < -self.x_threshold, x > self.x_threshold)
        theta_done = np.logical_or(theta < -self.theta_threshold_radians, theta > self.theta_threshold_radians)
        done = np.logical_or(x_done, theta_done).astype(np.int)
        return done

    def cost_fn_numpy_batch_v2(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        return np.abs(x) / self.x_threshold + np.abs(theta) / self.theta_threshold_radians

    def cost_fn_numpy_batch(self, states, actions, next_states):
        if self.version == 'v1':
            return self.cost_fn_numpy_batch_v1(states, actions, next_states)
        elif self.version == 'v2':
            return self.cost_fn_numpy_batch_v2(states, actions, next_states)
        else:
            raise NotImplementedError

    def cost_fn_torch_batch_v1(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        x_done = (x < -self.x_threshold) | (x > self.x_threshold)
        theta_done = (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)
        done = x_done | theta_done
        done = done.type(LongTensor)
        return done

    def cost_fn_torch_batch_v2(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        return torch.abs(x) / self.x_threshold + torch.abs(theta) / self.theta_threshold_radians

    def cost_fn_torch_batch(self, states, actions, next_states):
        if self.version == 'v1':
            return self.cost_fn_torch_batch_v1(states, actions, next_states)
        elif self.version == 'v2':
            return self.cost_fn_torch_batch_v2(states, actions, next_states)
        else:
            raise NotImplementedError


class ModelBasedPendulumWrapper(ModelBasedWrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = np.arctan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs

    def cost_fn_torch_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = torch.atan2(sin_th, cos_th)
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs


class ModelBasedRoboschoolInvertedPendulumWrapper(ModelBasedWrapper):
    def cost_fn_torch_batch(self, states, actions, next_states):
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = torch.atan2(sin_th, cos_th)
        done = torch.abs(theta) > .2
        return done.type(FloatTensor)

    def cost_fn_numpy_batch(self, states, actions, next_states):
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = np.arctan2(sin_th, cos_th)
        done = np.abs(theta) > .2
        return done.astype(np.float32)


class ModelBasedRoboschoolInvertedPendulumSwingupWrapper(ModelBasedWrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        return -next_states[:, 2]

    def cost_fn_torch_batch(self, states, actions, next_states):
        return -next_states[:, 2]


model_based_wrapper_dict = {
    'CartPole-v0': ModelBasedCartPoleWrapper,
    'CartPole-v1': ModelBasedCartPoleWrapper,
    'CartPoleContinuous-v0': ModelBasedCartPoleWrapper,
    'CartPoleContinuous-v1': ModelBasedCartPoleWrapper,
    'Pendulum-v0': ModelBasedPendulumWrapper,
    'PendulumNormalized-v0': ModelBasedPendulumWrapper,
    'RoboschoolInvertedPendulum-v1': ModelBasedRoboschoolInvertedPendulumWrapper,
    'RoboschoolInvertedPendulumSwingup-v1': ModelBasedRoboschoolInvertedPendulumSwingupWrapper,

}
