"""
Implement model-based reinforcement learning in https://arxiv.org/abs/1708.02596
Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
The steps are:
1. Collect random dataset (s, a, s', r) using random policy.
2. Train an initial dynamic model.
2. Fine-tune by using on policy data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Env
from tqdm import tqdm

from torchlib.common import convert_numpy_to_tensor, map_location
from torchlib.deep_rl import BaseAgent, RandomAgent
from torchlib.utils import normalize, unnormalize
from torchlib.utils.random.sampler import BaseSampler
from .utils import Dataset, gather_rollouts


class Agent(BaseAgent):
    def __init__(self, dynamics_model: nn.Module, optimizer,
                 action_sampler: BaseSampler,
                 cost_fn,
                 horizon=15,
                 num_random_action_selection=4096):
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_state_mean = None
        self.delta_state_std = None

        self.dynamics_model = dynamics_model
        self.optimizer = optimizer
        self.action_sampler = action_sampler
        self.horizon = horizon
        self.num_random_action_selection = num_random_action_selection
        self.cost_fn = cost_fn

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        states = {
            'dynamic_model': self.dynamics_model.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'delta_state_mean': self.delta_state_mean,
            'delta_state_std': self.delta_state_std
        }
        torch.save(states, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.dynamics_model.load_state_dict(states['dynamic_model'])
        self.state_mean = states['state_mean']
        self.state_std = states['state_std']
        self.action_mean = states['action_mean']
        self.action_std = states['action_std']
        self.delta_state_mean = states['delta_state_mean']
        self.delta_state_std = states['delta_state_std']

    def set_statistics(self, initial_dataset: Dataset):
        self.state_mean = convert_numpy_to_tensor(initial_dataset.state_mean)
        self.state_std = convert_numpy_to_tensor(initial_dataset.state_std)
        self.action_mean = convert_numpy_to_tensor(initial_dataset.action_mean)
        self.action_std = convert_numpy_to_tensor(initial_dataset.action_std)
        self.delta_state_mean = convert_numpy_to_tensor(initial_dataset.delta_state_mean)
        self.delta_state_std = convert_numpy_to_tensor(initial_dataset.delta_state_std)

    def predict(self, state):
        states = np.expand_dims(state, axis=0)
        actions = self.action_sampler.sample((self.horizon, self.num_random_action_selection)).astype(np.float32)
        states = np.tile(states, (self.num_random_action_selection, 1))
        states = convert_numpy_to_tensor(states)
        actions = convert_numpy_to_tensor(actions)

        with torch.no_grad():
            cost = 0
            for i in range(self.horizon):
                next_states = self.predict_next_states(states, actions[i])
                cost += self.cost_fn(states, actions[i], next_states)
                states = next_states

            best_action = actions[0, torch.argmin(cost, dim=0)]
            best_action = best_action.cpu().numpy()
            return best_action

    def predict_next_states(self, states, actions):
        assert self.state_mean is not None, 'Please set statistics before training for inference.'
        states = normalize(states, self.state_mean, self.state_std)
        actions = normalize(actions, self.action_mean, self.action_std)

        predicted_delta_state_normalized = self.dynamics_model.forward(states, actions)
        predicted_delta_state = unnormalize(predicted_delta_state_normalized, self.delta_state_mean,
                                            self.delta_state_std)
        return states + predicted_delta_state

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        t = range(epoch)
        if verbose:
            t = tqdm(t)

        for i in t:
            losses = []
            for states, actions, next_states, _, _ in dataset.random_iterator(batch_size=batch_size):
                # convert to tensor
                states = convert_numpy_to_tensor(states)
                actions = convert_numpy_to_tensor(actions)
                next_states = convert_numpy_to_tensor(next_states)
                # calculate loss
                self.optimizer.zero_grad()
                predicted_next_states = self.predict_next_states(states, actions)
                loss = F.mse_loss(predicted_next_states, next_states)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            t.set_description('Epoch {}/{}: Avg loss: {:.4f}'.format(i + 1, epoch, np.mean(losses)))


def train(env: Env, agent: Agent,
          num_init_random_rollouts=10,
          max_rollout_length=500,
          num_on_policy_iters=10,
          num_on_policy_rollouts=10,
          training_epochs=60,
          training_batch_size=512,
          verbose=True,
          checkpoint_path=None):
    # collect dataset using random policy
    random_policy = RandomAgent(env.action_space)
    dataset = gather_rollouts(env, random_policy, num_init_random_rollouts, max_rollout_length)

    agent.set_statistics(dataset)
    # train on initial dataset
    agent.fit_dynamic_model(dataset=dataset, epoch=training_epochs, batch_size=training_batch_size,
                            verbose=verbose)

    # gather new rollouts using MPC and retrain dynamics model
    for num_iter in range(num_on_policy_iters):
        if verbose:
            print('On policy iteration {}/{}'.format(num_iter + 1, num_on_policy_iters))
        on_policy_dataset = gather_rollouts(env, agent, num_on_policy_rollouts, max_rollout_length)

        # record on policy dataset statistics
        if verbose:
            stats = on_policy_dataset.log()
            strings = []
            for key, value in stats.items():
                strings.append(key + ": {:.4f}".format(value))
            strings = " - ".join(strings)
            print(strings)

        dataset.append(on_policy_dataset)

        agent.fit_dynamic_model(dataset=dataset, epoch=training_epochs, batch_size=training_batch_size,
                                verbose=verbose)

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)
