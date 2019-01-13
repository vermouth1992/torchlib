"""
Pytorch implementation of Vanilla PG. The code structure is adapted from UC Berkeley CS294-112.
Use various optimization techniques
1. Reward-to-go
2. Neural network baseline
3. Normalize advantage
4. Multiple threads to sample trajectory.
5. GAE-lambda
6. Multiple step update for PG
"""

import os

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.distributions import Categorical, MultivariateNormal
from torchlib.common import FloatTensor, eps, enable_cuda
from torchlib.deep_rl import BaseAgent

from .utils import compute_gae, compute_sum_of_rewards, sample_trajectories, pathlength


class Agent(BaseAgent):
    def __init__(self, policy_net: nn.Module, policy_optimizer, discrete, nn_baseline=True, lam=None,
                 value_coef=0.5):
        super(Agent, self).__init__()
        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.nn_baseline = nn_baseline
        self.baseline_loss = None if not nn_baseline else nn.MSELoss()
        self.discrete = discrete
        self.lam = lam
        self.value_coef = value_coef

    def get_action_distribution(self, state):
        if self.discrete:
            prob = self.policy_net.forward(state)[0]
            return Categorical(probs=prob)
        else:
            mean, logstd = self.policy_net.forward(state)[0]
            return MultivariateNormal(mean, torch.exp(logstd))

    def predict(self, state):
        """ Run the forward path of policy_network without gradient.

        Args:
            state: (batch_size, ob_dim)
            if discrete: probability distribution of a categorical distribution over actions
                sy_logits_na: (batch_size, self.ac_dim)
            if continuous: (mean, log_std) of a Gaussian distribution over actions
                sy_mean: (batch_size, self.ac_dim)
                sy_logstd: (self.ac_dim,)

        Returns:
            sy_sampled_ac:
                if discrete: (batch_size,)
                if continuous: (batch_size, self.ac_dim)

        """
        state = np.expand_dims(state, axis=0)
        with torch.no_grad():
            state = torch.from_numpy(state).type(FloatTensor)
            action = self.get_action_distribution(state).sample(torch.Size([])).cpu().numpy()
            return action[0]

    def get_baseline_loss(self, observation, rewards):
        # update baseline
        raw_baseline = self.policy_net.forward(observation)[1]
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + eps)
        rewards = rewards.type(FloatTensor)
        loss = self.baseline_loss(raw_baseline, rewards)
        return loss

    def construct_dataset(self, paths, gamma):
        rewards = compute_sum_of_rewards(paths, gamma)
        observation = np.concatenate([path["observation"] for path in paths])
        if self.nn_baseline:
            advantage = compute_gae(paths, gamma, self.policy_net, self.lam, np.mean(rewards), np.std(rewards))
        else:
            advantage = rewards

        # reshape all episodes to a single large batch
        actions = []
        for path in paths:
            actions.extend(path['actions'])

        # normalize advantage
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + eps)
        return actions, advantage, observation, rewards

    def update_policy(self, dataset, epoch=2):
        """ Update policy

        Args:
            paths: a list of trajectories. Each contain a list of symbolic log_prob and rewards

        Returns:

        """
        actions, advantage, observation, rewards = dataset

        observation = torch.Tensor(observation).type(FloatTensor)
        actions = torch.Tensor(actions)
        if enable_cuda:
            actions = actions.cuda()
        advantage = torch.Tensor(advantage).type(FloatTensor)
        rewards = torch.Tensor(rewards).type(FloatTensor)

        for _ in range(epoch):
            # update policy network
            self.policy_optimizer.zero_grad()
            # compute log prob
            distribution = self.get_action_distribution(observation)
            log_prob = distribution.log_prob(actions)

            assert log_prob.shape == advantage.shape, 'log_prob length {}, advantage length {}'.format(log_prob.shape,
                                                                                                       advantage.shape)

            action_loss = torch.mean(-log_prob * advantage)
            loss = action_loss

            if self.nn_baseline:
                value_loss = self.get_baseline_loss(observation, rewards)
                loss = loss + value_loss * self.value_coef

            loss.backward()
            self.policy_optimizer.step()

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.policy_net.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(state_dict)

    def predict_state_value(self, state):
        """ compute the state value using nn baseline

        Args:
            state: (batch_size, ob_dim)

        Returns: (batch_size,)

        """
        if not self.nn_baseline:
            raise ValueError('Baseline function is not defined')
        else:
            with torch.no_grad():
                return self.policy_net.forward(state)[1].cpu().numpy()


def train(exp, env, agent: Agent, n_iter, gamma, min_timesteps_per_batch, max_path_length,
          logdir=None, seed=1996, checkpoint_path=None):
    # Set random seeds
    env.seed(seed)
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    total_timesteps = 0

    if logdir:
        writer = SummaryWriter(log_dir=os.path.join(logdir, exp))
    else:
        writer = None

    best_avg_return = None

    for itr in range(n_iter):
        paths, timesteps_this_batch = sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length)

        total_timesteps += timesteps_this_batch

        datasets = agent.construct_dataset(paths, gamma)
        agent.update_policy(datasets)

        # logger
        returns = [np.sum(path["reward"]) for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)

        if best_avg_return is None or avg_return > best_avg_return:
            best_avg_return = avg_return
            if checkpoint_path:
                agent.save_checkpoint(checkpoint_path=checkpoint_path)

        if writer:
            writer.add_scalars('data/return', {'avg': avg_return,
                                               'std': std_return,
                                               'max': max_return,
                                               'min': min_return}, itr)
            writer.add_scalars('data/episode_length', {'avg': np.mean(ep_lengths),
                                                       'std': np.std(ep_lengths)}, itr)

        print('Iteration {} - Return {:.2f}±{:.2f} - Return range [{:.2f}, {:.2f}] - Best Avg Return {:.2f}'.format(
            itr, avg_return, std_return, min_return, max_return, best_avg_return))
