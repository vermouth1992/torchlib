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
from .policy import ImitationPolicy
from .utils import EpisodicDataset as Dataset, StateActionPairDataset


class VanillaAgent(BaseAgent):
    """
    In vanilla agent, it trains a world model and using the world model to plan.
    """

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
        self.model.eval()
        return self.planner.predict(state)

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        self.model.train()
        self.model.fit_dynamic_model(dataset, epoch, batch_size, verbose)

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        pass


class DAggerAgent(VanillaAgent):
    """
    Imitate optimal action by training a policy model using DAgger
    """

    def __init__(self, model, planner, policy: ImitationPolicy, policy_data_size=1000):
        super(DAggerAgent, self).__init__(model=model, planner=planner)
        self.policy = policy

        self.state_action_dataset = StateActionPairDataset(max_size=policy_data_size)

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        states = {
            'model': self.model.state_dict,
            'policy': self.policy.state_dict
        }
        torch.save(states, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(states['model'])
        self.policy.load_state_dict(states['policy'])

    def set_statistics(self, initial_dataset: Dataset):
        """ Set statistics for model and policy

        Args:
            initial_dataset: dataset collected by initial (random) policy

        Returns: None

        """
        super(DAggerAgent, self).set_statistics(initial_dataset=initial_dataset)
        self.policy.set_state_stats(initial_dataset.state_mean, initial_dataset.state_std)

    def predict(self, state):
        """ When collecting on policy data, we also bookkeeping optimal state, action pair
            (s, a) for training dagger model.

        Args:
            state: (state_dim,)

        Returns: (ac_dim,)

        """
        self.model.eval()
        action = self.planner.predict(state)
        self.state_action_dataset.add(state=state, action=action)
        self.policy.eval()
        action = self.policy.predict(state)
        return action

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        if len(self.state_action_dataset) > 0:
            self.policy.train()
            self.policy.set_state_stats(dataset.state_mean, dataset.state_std)
            self.policy.fit(self.state_action_dataset, epoch=epoch, batch_size=batch_size,
                            verbose=verbose)
