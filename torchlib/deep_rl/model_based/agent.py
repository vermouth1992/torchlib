"""
Implement model-based reinforcement learning in https://arxiv.org/abs/1708.02596
Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
The steps are:
1. Collect random dataset (s, a, s', r) using random policy.
2. Train an initial dynamic model.
2. Fine-tune by using on policy data.
"""

import torch

from torchlib.common import map_location
from torchlib.deep_rl import BaseAgent
from .model import Model
from .planner import Planner
from .utils import EpisodicDataset as Dataset


class VanillaAgent(BaseAgent):
    def __init__(self, model: Model, planner: Planner):
        self.model = model
        self.planner = planner

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.model.state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(states)

    def set_statistics(self, initial_dataset: Dataset):
        self.model.set_statistics(initial_dataset)

    def predict(self, state):
        return self.planner.predict(state)

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        self.model.fit_dynamic_model(dataset, epoch, batch_size, verbose)
