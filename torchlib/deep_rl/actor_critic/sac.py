"""
Soft actor critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
https://arxiv.org/pdf/1801.01290.pdf.
The advantage is that it automatically incorporates exploration by encouraging large entropy actions.
"""

import copy
import os

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from tqdm.auto import tqdm

from torchlib.common import FloatTensor, enable_cuda, convert_numpy_to_tensor
from torchlib.deep_rl import BaseAgent, RandomAgent
from torchlib.utils.random import set_global_seeds
from torchlib.utils.weight import soft_update, hard_update
from .utils import SimpleSampler, SimpleReplayPool


class SoftActorCritic(BaseAgent):
    def __init__(self,
                 policy_net: nn.Module, policy_optimizer,
                 q_network: nn.Module, q_optimizer,
                 discrete,
                 target_entropy=None,
                 alpha_optimizer=None,
                 log_alpha_tensor=None,
                 tau=0.01,
                 alpha=1.0,
                 discount=0.99,
                 ):
        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.discrete = discrete
        self.q_network = q_network
        self.q_optimizer = q_optimizer
        self.target_q_network = copy.deepcopy(self.q_network)

        hard_update(self.target_q_network, self.q_network)

        if log_alpha_tensor is not None:
            self._min_alpha = alpha
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
        obs = convert_numpy_to_tensor(obs)
        actions = convert_numpy_to_tensor(actions)
        next_obs = convert_numpy_to_tensor(next_obs)
        done = convert_numpy_to_tensor(done).type(FloatTensor)
        reward = convert_numpy_to_tensor(reward)

        # q loss
        q_values, q_values2 = self.q_network.forward(obs, actions, minimum=False)

        with torch.no_grad():
            next_action_distribution = self.policy_net.forward(next_obs)
            next_action = next_action_distribution.sample()
            target_q_values = self.target_q_network.forward(next_obs, next_action, minimum=True)
            q_target = reward + self._discount * (1.0 - done) * target_q_values

        q_values_loss = F.mse_loss(q_values, q_target) + F.mse_loss(q_values2, q_target)

        # policy loss
        if self.discrete:
            # for discrete action space, we can directly compute kl divergence analytically without sampling
            action_distribution = self.policy_net.forward(obs)
            q_values_min = self.q_network.forward(obs, minimum=True)  # (batch_size, ac_dim)
            probs = F.softmax(q_values_min, dim=-1)
            target_distribution = torch.distributions.Categorical(probs=probs)
            policy_loss = torch.distributions.kl_divergence(action_distribution, target_distribution).mean()

            pi = action_distribution.sample()
            log_prob = action_distribution.log_prob(pi)

        else:
            action_distribution = self.policy_net.forward(obs)
            pi, pre_tanh_pi = action_distribution.rsample(return_raw_value=True)
            log_prob = action_distribution.log_prob(pre_tanh_pi, is_raw_value=True)  # should be shape (batch_size,)
            q_values_pi, q_values2_pi = self.q_network.forward(obs, pi, minimum=False)
            q_values_pi_min = torch.min(q_values_pi, q_values2_pi)
            policy_loss = torch.mean(log_prob * self._alpha - q_values_pi_min)

        # alpha loss
        if self._log_alpha_tensor is not None:
            alpha_loss = -(self._log_alpha_tensor * (log_prob + self._target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._alpha = max(self._log_alpha_tensor.exp().item(), self._min_alpha)

            self.q_optimizer.zero_grad()
            q_values_loss.backward()
            self.q_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

    def get_alpha(self):
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
        observation_dtype=str(env.observation_space.dtype),
        action_dtype=str(env.action_space.dtype),
        max_size=replay_pool_size)

    sampler.initialize(env, RandomAgent(env.action_space), replay_pool)

    best_mean_episode_reward = -np.inf

    total_steps = prefill_steps

    for epoch in range(n_epochs):
        for _ in tqdm(range(epoch_length), desc='Epoch {}/{}'.format(epoch + 1, n_epochs)):
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
        last_period_episode_reward = episode_rewards[-100:]
        mean_episode_reward = np.mean(last_period_episode_reward)

        if mean_episode_reward > best_mean_episode_reward:
            if checkpoint_path:
                agent.save_checkpoint(checkpoint_path)

        std_episode_reward = np.std(last_period_episode_reward)
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        print("Total timesteps {}".format(total_steps))
        print("Mean reward (10 episodes) {:.2f}. std {:.2f}".format(np.mean(episode_rewards[-10:]),
                                                                    np.std(episode_rewards[-10:])))
        print("Mean reward (100 episodes) {:.2f}. std {:.2f}".format(mean_episode_reward, std_episode_reward))
        print('Reward range [{:.2f}, {:.2f}]'.format(np.min(last_period_episode_reward),
                                                     np.max(last_period_episode_reward)))
        print("Best mean reward {:.2f}".format(best_mean_episode_reward))
        print("Episodes {}".format(len(episode_rewards)))
        print("Alpha {:.4f}".format(agent.get_alpha()))
        print('------------')


def make_default_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_episode_length', type=int, default=1000)
    parser.add_argument('--no_automatic_entropy_tuning', action='store_true')
    parser.add_argument('--epoch_length', type=int, default=1000)
    parser.add_argument('--prefill_steps', type=int, default=1000)
    parser.add_argument('--replay_pool_size', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=123)
    return parser


def get_policy_net_q_network(env, args):
    """ Return the policy network and q network

    Args:
        env: standard gym env instance
        args: arguments

    Returns: policy network, q network

    """
    import gym
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    if len(env.observation_space.shape) == 1:
        # low dimensional environment
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        if discrete:
            from torchlib.deep_rl.models import DiscreteNNFeedForwardPolicy, DoubleQModule
            policy_net = DiscreteNNFeedForwardPolicy(nn_size=args['nn_size'], state_dim=ob_dim,
                                                     action_dim=ac_dim)
            q_network = DoubleQModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)
        else:
            from torchlib.deep_rl.models import ContinuousNNFeedForwardPolicy, DoubleCriticModule

            policy_net = ContinuousNNFeedForwardPolicy(nn_size=args['nn_size'], state_dim=ob_dim,
                                                       action_dim=ac_dim)
            q_network = DoubleCriticModule(size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim)

        return policy_net, q_network

    elif len(env.observation_space.shape) == 3:
        if env.observation_space.shape[:2] == (84, 84):
            # atari env
            from torchlib.deep_rl.models import AtariFeedForwardPolicy, DoubleAtariQModule
            policy_net = AtariFeedForwardPolicy(num_channel=args['frame_history_len'],
                                                action_dim=env.action_space.n)
            q_network = DoubleAtariQModule(frame_history_len=args['frame_history_len'],
                                           action_dim=env.action_space.n)
            return policy_net, q_network
        else:
            raise ValueError('Not a typical env. Please define custom network')

    else:
        raise ValueError('Not a typical env. Please define custom network')
