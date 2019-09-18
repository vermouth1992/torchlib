"""
Implement model-based reinforcement learning in https://arxiv.org/abs/1708.02596
Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
The steps are:
1. Collect random dataset (s, a, s', r) using random policy.
2. Train an initial dynamic model.
2. Fine-tune by using on policy data.
"""

import datetime
import time

import torch
from torchlib.common import map_location
from torchlib.deep_rl import BaseAgent, RandomAgent

from .environment import VirtualEnv
from .planner import Planner
from .policy import ImitationPolicy
from .utils import EpisodicDataset as Dataset, StateActionPairDataset, gather_rollouts
from .world_model import WorldModel
from ..policy_gradient import ppo


class ModelBasedAgent(BaseAgent):
    """
    In vanilla agent, it trains a world model and using the world model to plan.
    """

    def __init__(self, model: WorldModel):
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

    def train(self, env_fn, env=None,
              dataset_maxlen=10000,
              num_init_random_rollouts=10,
              max_rollout_length=1000,
              num_on_policy_iters=10,
              num_on_policy_rollouts=10,
              model_training_epochs=60,
              policy_training_epochs=60,
              training_batch_size=512,
              default_policy=None,
              verbose=True,
              checkpoint_path=None):
        """ Train model-based rl agent

        Args:
            env_fn: env fn for create parallel env
            env: a single env object. If not None, disable parallel env. Useful for env with memory.
            dataset_maxlen: data set size
            num_init_random_rollouts: Number of initial rollouts
            max_rollout_length: maximum rollout length
            num_on_policy_iters: Number of on-policy iterations
            num_on_policy_rollouts: Number of on-policy rollouts per iteration
            model_training_epochs: Model training epochs
            policy_training_epochs: policy training epochs
            training_batch_size: batch size
            default_policy: default policy to collect initial dataset
            verbose:
            checkpoint_path:

        Returns: None

        """

        start_time = time.time()

        # collect dataset using random policy
        single_env = env_fn() if env is None else env

        if default_policy is None:
            default_policy = RandomAgent(single_env.action_space)
        dataset = Dataset(maxlen=dataset_maxlen)

        print('Gathering initial dataset...')

        if max_rollout_length <= 0:
            max_rollout_length = single_env.spec.max_episode_steps

        initial_dataset = gather_rollouts(env_fn, env, default_policy, num_init_random_rollouts, max_rollout_length)
        dataset.append(initial_dataset)

        # gather new rollouts using MPC and retrain dynamics model
        for num_iter in range(num_on_policy_iters):
            if verbose:
                print('On policy iteration {}/{}. Size of dataset: {}. Number of trajectories: {}'.format(
                    num_iter + 1, num_on_policy_iters, len(dataset), dataset.num_trajectories))

            self.set_statistics(dataset)
            self.fit_dynamic_model(dataset=dataset, epoch=model_training_epochs, batch_size=training_batch_size,
                                   verbose=verbose)
            self.fit_policy(dataset=dataset, epoch=policy_training_epochs, batch_size=training_batch_size,
                            verbose=verbose)
            on_policy_dataset = gather_rollouts(env_fn, env, self, num_on_policy_rollouts, max_rollout_length)

            # record on policy dataset statistics

            if verbose:
                stats = on_policy_dataset.log()
                strings = []
                for key, value in stats.items():
                    strings.append(key + ": {:.4f}".format(value))
                strings = " - ".join(strings)
                print(strings)

            dataset.append(on_policy_dataset)

            time_elapse = datetime.timedelta(seconds=int(time.time() - start_time))
            print('Time elapsed {}'.format(time_elapse))

        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)


class ModelBasedPlanAgent(ModelBasedAgent):
    def __init__(self, model: WorldModel, planner: Planner):
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
            self.policy.fit(self.state_action_dataset, epoch=epoch, batch_size=batch_size,
                            verbose=verbose)


class ModelBasedPPOAgent(ModelBasedAgent):
    """
    Train model using real world interactions and update policy using PPO in simulated environments.
    """

    def __init__(self, model, ppo_agent: ppo.PPOAgent, real_env, **kwargs):
        super(ModelBasedPPOAgent, self).__init__(model=model)
        self.policy = ppo_agent
        self.virtual_env = VirtualEnv(model, real_env)

        self.gamma = kwargs.get('gamma', 0.99)
        self.min_timesteps_per_batch = kwargs.get('min_timesteps_per_batch', 1000)
        self.max_path_length = kwargs.get('max_path_length', None)
        if self.max_path_length is None or self.max_path_length <= 0:
            self.max_path_length = real_env.spec.max_episode_steps
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
        self.virtual_env.set_initial_states_pool(dataset.get_initial_states())
        self.model.eval()
        self.policy.train(exp=None, env=self.virtual_env, n_iter=epoch,
                          gamma=self.gamma, min_timesteps_per_batch=self.min_timesteps_per_batch,
                          max_path_length=self.max_path_length, logdir=None,
                          seed=self.seed, checkpoint_path=None)
