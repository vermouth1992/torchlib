"""
Policy Network for training imitation learning model. For discrete case, we use classifier.
For continuous case, we use regressor.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torchlib.common import move_tensor_to_gpu, convert_numpy_to_tensor, enable_cuda
from torchlib.deep_rl import BaseAgent
from torchlib.deep_rl.model_based.utils import StateActionPairDataset


class ImitationPolicy(BaseAgent):
    def __init__(self, model: nn.Module, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.state_mean = None
        self.state_std = None

        self.loss_fn = None

        if enable_cuda:
            self.model.cuda()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def state_dict(self):
        states = {
            'model': self.model.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }
        return states

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.state_mean = state_dict['state_mean']
        self.state_std = state_dict['state_std']

    def set_state_stats(self, state_mean, state_std):
        self.state_mean = convert_numpy_to_tensor(state_mean)
        self.state_std = convert_numpy_to_tensor(state_std)

    def predict(self, state):
        """

        Args:
            state: (ob_dim,)

        Returns:

        """
        raise NotImplementedError

    def fit(self, dataset: StateActionPairDataset, epoch=10, batch_size=128, verbose=False):
        t = range(epoch)
        if verbose:
            t = tqdm(t)

        for i in t:
            losses = []
            for state, action in dataset.random_iterator(batch_size=batch_size):
                self.optimizer.zero_grad()
                state = move_tensor_to_gpu(state)
                action = move_tensor_to_gpu(action)
                state = (state - self.state_mean) / self.state_std
                output = self.model.forward(state)
                loss = self.loss_fn(output, action)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if verbose:
                t.set_description('Epoch {}/{} - Avg policy loss: {:.4f}'.format(i + 1, epoch, np.mean(losses)))


class DiscretePolicy(ImitationPolicy):
    def __init__(self, model: nn.Module, optimizer):
        super(DiscretePolicy, self).__init__(model=model, optimizer=optimizer)
        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        with torch.no_grad():
            state = convert_numpy_to_tensor(state)
            state = (state - self.state_mean) / self.state_std
            action = self.model.forward(state)
            action = torch.argmax(action, dim=-1)
        return action.cpu().numpy()[0]


class ContinuousPolicy(ImitationPolicy):
    """
    For continuous policy, we assume the action space is between -1 and 1.
    So we use tanh as final activation layer.
    """

    def __init__(self, model: nn.Module, optimizer):
        super(ContinuousPolicy, self).__init__(model=model, optimizer=optimizer)
        self.loss_fn = nn.MSELoss()

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        with torch.no_grad():
            state = convert_numpy_to_tensor(state)
            state = (state - self.state_mean) / self.state_std
            action = self.model.forward(state)
        return action.cpu().numpy()[0]
