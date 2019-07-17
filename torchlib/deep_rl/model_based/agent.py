"""
Implement model-based reinforcement learning in https://arxiv.org/abs/1708.02596
Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
The steps are:
1. Collect random dataset (s, a, s', r) using random policy.
2. Train an initial dynamic model.
2. Fine-tune by using on policy data.
"""

import torch

import torchlib.deep_rl.policy_gradient as pg
from torchlib.common import map_location
from torchlib.deep_rl import BaseAgent
from .environment import VirtualEnv
from .model import Model
from .planner import Planner
from .policy import ImitationPolicy
from .utils import EpisodicDataset as Dataset, StateActionPairDataset


class ModelBasedAgent(BaseAgent):
    """
    In vanilla agent, it trains a world model and using the world model to plan.
    """

    def __init__(self, model: Model):
        self.model = model

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.model.state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(states)

    def set_statistics(self, initial_dataset: Dataset):
        self.model.set_statistics(initial_dataset)

    def predict(self, state):
        raise NotImplementedError

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        self.model.train()
        self.model.fit_dynamic_model(dataset, epoch, batch_size, verbose)

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        raise NotImplementedError


class ModelBasedPlanAgent(ModelBasedAgent):
    def __init__(self, model: Model, planner: Planner):
        super(ModelBasedPlanAgent, self).__init__(model=model)
        self.planner = planner

    def predict(self, state):
        self.model.eval()
        return self.planner.predict(state)

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        pass


class ModelBasedDAggerAgent(ModelBasedPlanAgent):
    """
    Imitate optimal action by training a policy model using DAgger
    """

    def __init__(self, model, planner, policy: ImitationPolicy, policy_data_size=1000):
        super(ModelBasedDAggerAgent, self).__init__(model=model, planner=planner)
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
        super(ModelBasedDAggerAgent, self).set_statistics(initial_dataset=initial_dataset)
        self.policy.set_state_stats(initial_dataset.state_mean, initial_dataset.state_std)

    def predict(self, state):
        """ When collecting on policy data, we also bookkeeping optimal state, action pair
            (s, a) for training dagger model.

        Args:
            state: (state_dim,)

        Returns: (ac_dim,)

        """
        action = super(ModelBasedDAggerAgent, self).predict(state=state)
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


class ModelBasedPPOAgent(ModelBasedAgent):
    """
    Train model using real world interactions and update policy using PPO in simulated environments.
    """

    def __init__(self, model, ppo_agent: pg.PPOAgent, real_env, **kwargs):
        super(ModelBasedPPOAgent, self).__init__(model=model)
        self.policy = ppo_agent
        self.world_model = VirtualEnv(model, real_env)

        self.gamma = kwargs.get('gamma', 0.99)
        self.min_timesteps_per_batch = kwargs.get('min_timesteps_per_batch', 1000)
        self.max_path_length = kwargs.get('max_path_length', 1000)
        self.seed = kwargs.get('seed', 1996)

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

    def predict(self, state):
        return self.policy.predict(state)

    def fit_policy(self, dataset: Dataset, epoch=5, batch_size=128, verbose=True):
        self.world_model.set_initial_states_pool(dataset.get_initial_states())
        pg.train(exp=None, env=self.world_model, agent=self.policy, n_iter=epoch,
                 gamma=self.gamma, min_timesteps_per_batch=self.min_timesteps_per_batch,
                 max_path_length=self.max_path_length, logdir=None,
                 seed=self.seed, checkpoint_path=None)
