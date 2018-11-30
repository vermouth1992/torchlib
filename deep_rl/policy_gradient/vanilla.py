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
from torch.distributions import Categorical, Normal
from torchlib.common import FloatTensor, eps
from torchlib.utils.random.torch_random_utils import set_global_seeds


class Agent(object):
    def __init__(self, policy_net: nn.Module, policy_optimizer, discrete, nn_baseline=None,
                 nn_baseline_optimizer=None):
        super(Agent, self).__init__()
        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.nn_baseline = nn_baseline
        self.baseline_loss = None if not self.nn_baseline else nn.MSELoss()
        if self.nn_baseline:
            assert nn_baseline_optimizer is not None, "nn_baseline_optimizer can' be None"
        self.nn_baseline_optimizer = nn_baseline_optimizer
        self.discrete = discrete

    def get_action_distribution(self, state):
        state = torch.from_numpy(state).type(FloatTensor)
        if self.discrete:
            probs = self.policy_net.forward(state)
            return Categorical(probs=probs)
        else:
            mean, logstd = self.policy_net.forward(state)
            return Normal(mean, torch.exp(logstd))

    def sample_action(self, state):
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
        with torch.no_grad():
            batch_size = state.shape[0]
            state = torch.from_numpy(state).type(FloatTensor)
            if self.discrete:
                prob = self.policy_net.forward(state)
                return Categorical(prob).sample(batch_size)
            else:
                mean, logstd = self.policy_net.forward(state)
                return Normal(mean, torch.exp(logstd)).sample(batch_size)

    def update_policy(self, observation, log_prob, rewards, num_trajectories):
        """ Update policy

        Args:
            paths: a list of trajectories. Each contain a list of symbolic log_prob and rewards

        Returns:

        """
        if self.nn_baseline:
            raw_baseline = self.nn_baseline.forward(observation)

            # we assume baseline data has mean of 0 and std of 1.
            raw_baseline_data = raw_baseline.detach().cpu().numpy()
            raw_baseline_data = raw_baseline_data * np.std(rewards) + np.mean(rewards)
            advantage = rewards - raw_baseline_data

            # update nn baseline
            self.nn_baseline_optimizer.zero_grad()
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)
            rewards = torch.tensor(rewards).type(FloatTensor)
            loss = self.baseline_loss(raw_baseline, rewards)
            loss.backward()
            self.nn_baseline_optimizer.step()
        else:
            advantage = rewards

        # normalize advantage
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + eps)
        advantage = torch.from_numpy(advantage).type(FloatTensor)

        policy_loss = []
        assert len(log_prob) == len(advantage), 'log_prob length {}, advantage length {}'.format(len(log_prob),
                                                                                                 len(advantage))
        for i in range(len(log_prob)):
            policy_loss.append(-log_prob[i] * advantage[i])

        # update policy network
        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() / num_trajectories
        policy_loss.backward()
        self.policy_optimizer.step()

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.policy_net.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(state_dict)


def pathlength(path):
    return len(path["reward"])


def compute_sum_of_rewards(paths, gamma):
    # calculate sum of reward
    rewards = []
    for path in paths:
        current_rewards = []
        # calculate reward-to-go
        R = 0
        for r in path['reward'][::-1]:
            R = r + gamma * R
            current_rewards.insert(0, R)
        rewards.extend(current_rewards)
    rewards = np.array(rewards)
    return rewards


def sample_trajectory(agent, env, max_path_length):
    ob = env.reset()
    log_prob, rewards, obs = [], [], []
    steps = 0
    while True:
        distribution = agent.get_action_distribution(np.array(ob))
        # ac and log_prob are nodes on computational graph
        ac = distribution.sample()
        log_prob.append(distribution.log_prob(ac))
        obs.append(ob)

        ob, rew, done, _ = env.step(ac.cpu().numpy())
        rewards.append(rew)
        steps += 1
        if done or steps > max_path_length:
            break
    path = {"log_prob": log_prob,
            "reward": rewards,
            "observation": np.array(obs)}
    return path


def sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length):
    timesteps_this_batch = 0
    paths = []
    while True:
        path = sample_trajectory(agent, env, max_path_length)
        paths.append(path)
        timesteps_this_batch += pathlength(path)
        if timesteps_this_batch > min_timesteps_per_batch:
            break
    return paths, timesteps_this_batch


def train(exp, env, agent: Agent, n_iter, gamma, min_timesteps_per_batch, max_path_length,
          logdir='runs', seed=1996):
    # Set random seeds
    set_global_seeds(seed)
    env.seed(seed)
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    total_timesteps = 0

    writer = SummaryWriter(log_dir=os.path.join(logdir, exp))

    for itr in range(n_iter):
        paths, timesteps_this_batch = sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length)

        observation = torch.from_numpy(np.concatenate([path["observation"] for path in paths])).type(FloatTensor)
        log_prob = []
        for path in paths:
            log_prob.extend(path['log_prob'])

        rewards = compute_sum_of_rewards(paths, gamma)

        total_timesteps += timesteps_this_batch
        agent.update_policy(observation, log_prob, rewards, len(paths))

        # logger
        returns = [np.sum(path["reward"]) for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        avg_return = np.mean(returns)
        max_return = np.max(returns)
        writer.add_scalars('data/return', {'avg': avg_return,
                                           'std': np.std(returns),
                                           'max': max_return,
                                           'min': np.min(returns)}, itr)
        writer.add_scalars('data/episode_length', {'avg': np.mean(ep_lengths),
                                                   'std': np.std(ep_lengths)}, itr)

        print('Iteration {} - Avg Return {:.2f} - Max Return {:.2f}'.format(itr, avg_return, max_return))
