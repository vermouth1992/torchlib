"""
Models for model-based RL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchlib.common import enable_cuda, move_tensor_to_gpu, convert_numpy_to_tensor
from torchlib.utils import normalize, unnormalize
from .utils import EpisodicDataset as Dataset


class Model(object):
    def set_statistics(self, initial_dataset):
        raise NotImplementedError

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        raise NotImplementedError

    def predict_next_state(self, state, action):
        states = np.expand_dims(state, axis=0)
        actions = np.expand_dims(action, axis=0)
        states = convert_numpy_to_tensor(states)
        actions = convert_numpy_to_tensor(actions)
        with torch.no_grad():
            next_state = self.predict_next_states(states, actions).cpu().numpy()[0]
        return next_state

    def predict_next_states(self, states, actions):
        raise NotImplementedError

    def cost_fn(self, states, actions):
        raise NotImplementedError

    @property
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, states):
        raise NotImplementedError


class DeterministicModel(Model):
    """
    deterministic model following equation s_{t+1} = s_{t} + f(s_{t}, a_{t})
    """

    def __init__(self, dynamics_model: nn.Module, optimizer):
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_state_mean = None
        self.delta_state_std = None

        self.dynamics_model = dynamics_model
        self.optimizer = optimizer

        if enable_cuda:
            self.dynamics_model.cuda()

    def set_statistics(self, initial_dataset: Dataset):
        self.state_mean = convert_numpy_to_tensor(initial_dataset.state_mean)
        self.state_std = convert_numpy_to_tensor(initial_dataset.state_std)
        if self.dynamics_model.discrete:
            self.action_mean = None
            self.action_std = None
        else:
            self.action_mean = convert_numpy_to_tensor(initial_dataset.action_mean)
            self.action_std = convert_numpy_to_tensor(initial_dataset.action_std)
        self.delta_state_mean = convert_numpy_to_tensor(initial_dataset.delta_state_mean)
        self.delta_state_std = convert_numpy_to_tensor(initial_dataset.delta_state_std)

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        t = range(epoch)
        if verbose:
            t = tqdm(t)

        for i in t:
            losses = []
            for states, actions, next_states, _, _ in dataset.random_iterator(batch_size=batch_size):
                # convert to tensor
                states = move_tensor_to_gpu(states)
                actions = move_tensor_to_gpu(actions)
                next_states = move_tensor_to_gpu(next_states)
                # calculate loss
                self.optimizer.zero_grad()
                predicted_next_states = self.predict_next_states(states, actions)
                loss = F.mse_loss(predicted_next_states, next_states)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if verbose:
                t.set_description('Epoch {}/{} - Avg model loss: {:.4f}'.format(i + 1, epoch, np.mean(losses)))

    def predict_next_states(self, states, actions):
        assert self.state_mean is not None, 'Please set statistics before training for inference.'
        states_normalized = normalize(states, self.state_mean, self.state_std)

        if not self.dynamics_model.discrete:
            actions = normalize(actions, self.action_mean, self.action_std)

        predicted_delta_state_normalized = self.dynamics_model.forward(states_normalized, actions)
        predicted_delta_state = unnormalize(predicted_delta_state_normalized, self.delta_state_mean,
                                            self.delta_state_std)
        return states + predicted_delta_state

    def state_dict(self):
        states = {
            'dynamic_model': self.dynamics_model.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'delta_state_mean': self.delta_state_mean,
            'delta_state_std': self.delta_state_std
        }
        return states

    def load_state_dict(self, states):
        self.dynamics_model.load_state_dict(states['dynamic_model'])
        self.state_mean = states['state_mean']
        self.state_std = states['state_std']
        self.action_mean = states['action_mean']
        self.action_std = states['action_std']
        self.delta_state_mean = states['delta_state_mean']
        self.delta_state_std = states['delta_state_std']
