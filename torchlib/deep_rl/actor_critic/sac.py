"""
Soft actor critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
https://arxiv.org/pdf/1801.01290.pdf.
The advantage is that it automatically incorporates exploration by encouraging large entropy actions.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchlib.common import FloatTensor, enable_cuda
from torchlib.deep_rl import BaseAgent
from torchlib.utils.random import set_global_seeds
from torchlib.utils.weight import soft_update, hard_update
from tqdm import tqdm

from .utils import SimpleSampler, SimpleReplayPool


class SoftActorCritic(BaseAgent):
    def __init__(self,
                 policy_net: nn.Module, policy_optimizer,
                 q_network: nn.Module, q_optimizer,
                 target_entropy=None,
                 alpha_optimizer=None,
                 log_alpha_tensor=None,
                 tau=0.01,
                 alpha=1.0,
                 discount=0.99,
                 ):
        """

        Args:
            policy_net: return a distribution
            policy_optimizer:
            q_network:
            q_network2:
            q_optimizer:
            value_network:
            value_optimizer:
        """

        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.q_network = q_network
        self.q_optimizer = q_optimizer
        self.target_q_network = copy.deepcopy(self.q_network)

        hard_update(self.target_q_network, self.q_network)

        self._alpha = alpha
        self._discount = discount
        self._tau = tau

        self._target_entropy = target_entropy
        self._log_alpha_tensor = log_alpha_tensor
        self.alpha_optimizer = alpha_optimizer

        if enable_cuda:
            self.policy_net.cuda()
            self.q_network.cuda()
            self.target_q_network.cuda()

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self._tau)

    def update(self, obs, actions, next_obs, done, reward):
        """ Sample a mini-batch from replay buffer and update the network

        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)

        Returns: None

        """
        obs = torch.from_numpy(obs).type(FloatTensor)
        actions = torch.from_numpy(actions).type(FloatTensor)
        next_obs = torch.from_numpy(next_obs).type(FloatTensor)
        done = torch.from_numpy(done).type(FloatTensor)
        reward = torch.from_numpy(reward).type(FloatTensor)

        # q loss

        q_values, q_values2 = self.q_network.forward(obs, actions)

        with torch.no_grad():
            next_action_distribution = self.policy_net.forward(next_obs)
            next_action = next_action_distribution.sample()
            target_q_values, target_q_values2 = self.target_q_network.forward(next_obs, next_action)
            target_q_values = torch.min(target_q_values, target_q_values2)
            q_target = reward + self._discount * (1.0 - done) * target_q_values

        q_values_loss = F.mse_loss(q_values, q_target) + F.mse_loss(q_values2, q_target)

        # policy loss

        action_distribution = self.policy_net.forward(obs)

        pi, pre_tanh_pi = action_distribution.rsample(return_raw_value=True)

        log_prob = action_distribution.log_prob(pre_tanh_pi, is_raw_value=True)  # should be shape (batch_size,)

        q_values_pi, q_values2_pi = self.q_network.forward(obs, pi)

        q_values_pi_min = torch.min(q_values_pi, q_values2_pi)

        policy_loss = torch.mean(log_prob * self._alpha - q_values_pi_min)

        # alpha loss
        if self._log_alpha_tensor is not None:
            alpha_loss = -(self._log_alpha_tensor * (log_prob + self._target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._alpha = self._log_alpha_tensor.exp()

        self.q_optimizer.zero_grad()
        q_values_loss.backward()
        self.q_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def get_alpha(self):
        if self._log_alpha_tensor is not None:
            return self._alpha.item()
        else:
            return self._alpha

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        with torch.no_grad():
            obs = torch.from_numpy(state).type(FloatTensor)
            action_distribution = self.policy_net.forward(obs)
            actions = action_distribution.sample().cpu().numpy()
            return actions[0]

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.policy_net.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(state_dict)


def train(exp, env, agent: SoftActorCritic, n_epochs, max_episode_length, prefill_steps, epoch_length,
          replay_pool_size, batch_size, seed, checkpoint_path=None):
    from gym import wrappers
    from torchlib.deep_rl.envs.wrappers import get_wrapper_by_name

    set_global_seeds(seed)
    env.seed(seed)

    # create a Monitor for env
    expt_dir = '/tmp/{}'.format(exp)
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

    sampler = SimpleSampler(max_episode_length=max_episode_length, prefill_steps=prefill_steps)
    replay_pool = SimpleReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        max_size=replay_pool_size)

    sampler.initialize(env, agent, replay_pool)

    best_mean_episode_reward = -np.inf

    total_steps = prefill_steps

    for epoch in range(n_epochs):
        for _ in tqdm(range(epoch_length)):
            sampler.sample()

            batch = sampler.random_batch(batch_size)

            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']
            reward = batch['rewards']
            done = batch['terminals']

            agent.update(obs=obs, actions=actions, next_obs=next_obs, done=done, reward=reward)
            agent.update_target()

            total_steps += 1

        # logging
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        last_period_episode_reward = episode_rewards[-10:]
        mean_episode_reward = np.mean(last_period_episode_reward)
        print('------------')
        if mean_episode_reward > best_mean_episode_reward:
            if checkpoint_path:
                agent.save_checkpoint(checkpoint_path)

        std_episode_reward = np.std(last_period_episode_reward)
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        print("Epoch {}/{}. Total timesteps {}".format(epoch + 1, n_epochs, total_steps))
        print("mean reward (10 episodes) {:.2f}. std {:.2f}".format(mean_episode_reward, std_episode_reward))
        print('reward range [{:.2f}, {:.2f}]'.format(np.min(last_period_episode_reward),
                                                     np.max(last_period_episode_reward)))
        print("best mean reward {:.2f}".format(best_mean_episode_reward))
        print("episodes %d" % len(episode_rewards))
        print("alpha {:.2f}".format(agent.get_alpha()))


def make_default_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_episode_length', type=int, default=1000)
    parser.add_argument('--no_automatic_entropy_tuning', action='store_true')
    parser.add_argument('--epoch_length', type=int, default=1000)
    parser.add_argument('--prefill_steps', type=int, default=1000)
    parser.add_argument('--replay_pool_size', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=123)
    return parser
