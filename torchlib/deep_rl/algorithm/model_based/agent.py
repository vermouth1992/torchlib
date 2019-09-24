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
from torchlib.common import map_location
from torchlib.deep_rl import BaseAgent, RandomAgent
from torchlib.deep_rl.utils.replay.replay import TransitionReplayBuffer
from torchlib.deep_rl.utils.replay.sampler import StepSampler
from torchlib.utils.logx import EpochLogger
from torchlib.utils.timer import Timer

from .planner import Planner
from .policy import ImitationPolicy
from .utils import StateActionPairDataset
from .world_model import WorldModel


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

    def set_statistics(self, initial_dataset):
        self.model.set_statistics(initial_dataset)

    def predict_batch(self, states):
        raise NotImplementedError

    def fit_dynamic_model(self, dataset, epoch=10, batch_size=128, logger=None):
        self.model.train()
        self.model.fit_dynamic_model(dataset, epoch, batch_size, logger)

    def fit_policy(self, dataset, epoch=10, batch_size=128, logger=None):
        raise NotImplementedError

    def train(self, env, exp_name,
              replay_pool_size=1000000,
              prefill_steps=1000,
              num_epochs=100,
              epoch_length=2000,
              model_training_epochs=60,
              policy_training_epochs=60,
              training_batch_size=512,
              default_policy=None,
              log_dir=None,
              checkpoint_path=None):
        """ Train model-based RL agent

        Args:
            env: vector OpenAI gym environment
            replay_pool_size: replay pool size for training model-based RL
            prefill_steps:
            num_epochs:
            epoch_length:
            model_training_epochs:
            policy_training_epochs:
            training_batch_size:
            default_policy:
            log_dir:
            checkpoint_path:

        Returns:

        """

        timer = Timer()
        best_mean_episode_reward = -np.inf
        if default_policy is None:
            default_policy = RandomAgent(env.action_space)

        logger = EpochLogger(output_dir=log_dir, exp_name=exp_name)
        dataset = TransitionReplayBuffer(
            capacity=replay_pool_size,
            obs_shape=env.single_observation_space.shape,
            obs_dtype=env.single_observation_space.dtype,
            ac_shape=env.single_action_space.shape,
            ac_dtype=env.single_action_space.dtype,
        )
        sampler = StepSampler(prefill_steps=prefill_steps, logger=logger)
        sampler.initialize(env, default_policy, dataset)
        total_timesteps = prefill_steps // env.num_envs * prefill_steps

        # gather new rollouts using MPC and retrain dynamics model
        for epoch in range(num_epochs):
            avg_return = logger.get_stats('EpReward')[0]
            if avg_return > best_mean_episode_reward:
                best_mean_episode_reward = avg_return
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)

            # logging
            logger.log_tabular('On policy iteration (total {})'.format(num_epochs), epoch + 1)
            logger.log_tabular('Time Elapsed', timer.get_time_elapsed())
            logger.log_tabular('EpReward', with_min_and_max=True)
            logger.log_tabular('EpLength', average_only=True, with_min_and_max=True)
            logger.log_tabular('TotalSteps', total_timesteps)
            logger.log_tabular('TotalEpisodes', sampler.get_total_episode())
            logger.log_tabular('ModelTrainLoss', average_only=True)
            logger.log_tabular('ModelValLoss', average_only=True)
            logger.log_tabular('BestAvgReward', best_mean_episode_reward)
            logger.log_tabular('Replay Size', len(dataset))
            logger.dump_tabular()

            self.set_statistics(dataset)
            self.fit_dynamic_model(dataset=dataset, epoch=model_training_epochs, batch_size=training_batch_size,
                                   logger=logger)
            self.fit_policy(dataset=dataset, epoch=policy_training_epochs, batch_size=training_batch_size,
                            logger=logger)

            for _ in range(epoch_length):
                sampler.sample(policy=self)

            total_timesteps += epoch_length * env.num_envs


class ModelBasedPlanAgent(ModelBasedAgent):
    def __init__(self, model: WorldModel, planner: Planner):
        super(ModelBasedPlanAgent, self).__init__(model=model)
        self.planner = planner

    def predict(self, states):
        self.model.eval()
        return self.planner.predict_batch(states)

    def fit_policy(self, dataset, epoch=10, batch_size=128, verbose=False):
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

    def set_statistics(self, initial_dataset):
        """ Set statistics for model and policy

        Args:
            initial_dataset: dataset collected by initial (random) policy

        Returns: None

        """
        super(ModelBasedDAggerAgent, self).set_statistics(initial_dataset=initial_dataset)
        self.policy.set_state_stats(*initial_dataset.state_mean_std)

    def predict_batch(self, states):
        actions = super(ModelBasedDAggerAgent, self).predict_batch(states=states)
        self.state_action_dataset.add(state=states, action=actions)
        self.policy.eval()
        action = self.policy.predict_batch(states)
        return action

    def fit_policy(self, dataset, epoch=10, batch_size=128, verbose=False):
        if len(self.state_action_dataset) > 0:
            self.policy.train()
            self.policy.fit(self.state_action_dataset, epoch=epoch, batch_size=batch_size,
                            verbose=verbose)
