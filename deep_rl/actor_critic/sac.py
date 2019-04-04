"""
Soft actor critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
https://arxiv.org/pdf/1801.01290.pdf.
The advantage is that it automatically incorporates exploration by encouraging large entropy actions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.common import FloatTensor
from torchlib.deep_rl import BaseAgent
from torchlib.utils.random.torch_random_utils import set_global_seeds
from .utils import SimpleSampler, SimpleReplayPool
import copy

class SoftActorCritic(BaseAgent):
    def __init__(self,
                 policy_net: nn.Module,
                 q_network: nn.Module, q_network2: nn.Module,
                 value_network: nn.Module,
                 optimizer,
                 tau=0.01,
                 reparameterize=False,
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
        self.q_network = q_network
        self.q_network2 = q_network2
        self.value_network = value_network
        self.optimizer = optimizer
        self.target_value_network = copy.deepcopy(value_network)

        self._reparameterize = reparameterize
        self._alpha = alpha
        self._discount = discount
        self._tau = tau

    def _compute_policy_loss(self, obs):
        obs = torch.from_numpy(obs).type(FloatTensor)
        action_distribution = self.policy_net.forward(obs)
        actions = action_distribution.sample(torch.Size([obs.shape[0]]))

        q_values = self.q_network.forward(obs, actions)
        q_values2 = self.q_network2.forward(obs, actions)
        q_values = torch.min(q_values, q_values2)  # should be shape (batch_size,)

        log_prob = action_distribution.log_prob(actions)  # should be shape (batch_size,)
        q_values = q_values.detach()

        if self._reparameterize:
            policy_loss = torch.mean(log_prob * self._alpha - q_values)

        else:
            values = self.value_network.forward(obs).detach()  # should be shape (batch_size,)
            log_prob_detached = log_prob.detach()

            policy_loss = torch.mean(log_prob * (self._alpha * log_prob_detached - q_values + values))

        return policy_loss

    def _compute_value_loss(self, obs):
        obs = torch.from_numpy(obs).type(FloatTensor)
        action_distribution = self.policy_net.forward(obs)
        actions = action_distribution.sample(torch.Size([obs.shape[0]]))

        q_values = self.q_network.forward(obs, actions)
        q_values2 = self.q_network2.forward(obs, actions)
        q_values = torch.min(q_values, q_values2)  # should be shape (batch_size,)

        expected_values = q_values - self._alpha * action_distribution.log_prob(actions)

        values = self.value_network.forward(obs)  # should be shape (batch_size,)

        loss = F.mse_loss(values, expected_values.detach())

        return loss

    def _compute_q_loss(self, obs, actions, next_obs, done, reward):
        obs = torch.from_numpy(obs).type(FloatTensor)
        actions = torch.from_numpy(actions).type(FloatTensor)
        next_obs = torch.from_numpy(next_obs).type(FloatTensor)
        done = torch.from_numpy(done).type(FloatTensor)
        reward = torch.from_numpy(reward).type(FloatTensor)

        target_values = self.target_value_network.forward(next_obs)
        q_values = self.q_network.forward(obs, actions)
        q_values2 = self.q_network2.forward(obs, actions)
        target_q_values = reward + self._discount * (1.0 - done) * target_values
        target_q_values = target_q_values.detach()
        loss = F.mse_loss(q_values, target_q_values) + F.mse_loss(q_values2, target_q_values)
        return loss

    def update_target(self):
        source = self.value_network
        target = self.target_value_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau
            )

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
        self.optimizer.zero_grad()

        policy_loss = self._compute_policy_loss(obs)
        value_loss = self._compute_value_loss(obs)
        q_loss = self._compute_q_loss(obs, actions, next_obs, done, reward)
        losses = [policy_loss, value_loss, q_loss]
        loss = sum(losses)
        loss.backward()

        self.optimizer.step()


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

    best_mean_episode_reward = -1

    for epoch in range(n_epochs):
        for t in range(epoch_length):
            sampler.sample()

            batch = sampler.random_batch(batch_size)

            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']
            reward = batch['rewards']
            done = batch['terminals']

            agent.update(obs=obs, actions=actions, next_obs=next_obs, done=done, reward=reward)
            agent.update_target()

        # logging
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        last_one_hundred_episode_reward = episode_rewards[-100:]
        mean_episode_reward = np.mean(last_one_hundred_episode_reward)
        print('------------')
        if mean_episode_reward > best_mean_episode_reward:
            if checkpoint_path:
                agent.save_checkpoint(checkpoint_path)

        std_episode_reward = np.std(last_one_hundred_episode_reward)
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        print("Epoch {}/{}. Total timesteps {}".format(epoch + 1, n_epochs,
                                                       sampler.get_statistics()['TimestepsSoFar']))
        print("mean reward (100 episodes) {:.2f}. std {:.2f}".format(mean_episode_reward, std_episode_reward))
        print('reward range [{:.2f}, {:.2f}]'.format(np.min(last_one_hundred_episode_reward),
                                                     np.max(last_one_hundred_episode_reward)))
        print("best mean reward {:.2f}".format(best_mean_episode_reward))
        print("episodes %d" % len(episode_rewards))


def make_default_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--reparameterize', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    return parser


