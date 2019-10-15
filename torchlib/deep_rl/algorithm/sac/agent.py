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
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
from torchlib.common import FloatTensor, enable_cuda, convert_to_tensor, device
from torchlib.deep_rl import BaseAgent
from torchlib.deep_rl.utils.replay.replay import TransitionReplayBuffer
from torchlib.deep_rl.utils.replay.sampler import StepSampler
from torchlib.utils.logx import EpochLogger
from torchlib.utils.timer import Timer
from torchlib.utils.weight import soft_update, hard_update
from tqdm.auto import tqdm


class Agent(BaseAgent):
    def __init__(self,
                 nets,
                 discrete,
                 learning_rate=3e-4,
                 target_entropy=None,
                 alpha=1.0,
                 tau=5e-3,
                 gamma=0.99,
                 **kwargs,
                 ):
        self.policy_net = nets['policy_net']
        self.q_network = nets['q_network']
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.discrete = discrete
        self.target_q_network = copy.deepcopy(self.q_network)

        hard_update(self.target_q_network, self.q_network)

        if target_entropy is not None:
            self.log_alpha_tensor = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha_tensor], lr=learning_rate)
            self.target_entropy = target_entropy
        else:
            self.log_alpha_tensor = None

        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma

        if enable_cuda:
            self.policy_net.cuda()
            self.q_network.cuda()
            self.target_q_network.cuda()

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

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
        obs = convert_to_tensor(obs)
        actions = convert_to_tensor(actions)
        next_obs = convert_to_tensor(next_obs)
        done = convert_to_tensor(done).type(FloatTensor)
        reward = convert_to_tensor(reward)

        # q loss
        q_values, q_values2 = self.q_network.forward(obs, actions, False)

        with torch.no_grad():
            next_action_distribution = self.policy_net.forward_action(next_obs)
            next_action = next_action_distribution.sample()
            next_action_log_prob = next_action_distribution.log_prob(next_action)
            target_q_values = self.target_q_network.forward(next_obs, next_action,
                                                            True) - self.alpha * next_action_log_prob
            q_target = reward + self.gamma * (1.0 - done) * target_q_values

        q_values_loss = F.mse_loss(q_values, q_target) + F.mse_loss(q_values2, q_target)

        # policy loss
        if self.discrete:
            # for discrete action space, we can directly compute kl divergence analytically without sampling
            action_distribution = self.policy_net.forward_action(obs)
            q_values_min = self.q_network.forward(obs, None, True)  # (batch_size, ac_dim)
            probs = F.softmax(q_values_min, dim=-1)
            target_distribution = torch.distributions.Categorical(probs=probs)
            policy_loss = torch.distributions.kl_divergence(action_distribution, target_distribution).mean()
            log_prob = -action_distribution.entropy()

        else:
            action_distribution = self.policy_net.forward_action(obs)
            pi = action_distribution.rsample()
            log_prob = action_distribution.log_prob(pi)  # should be shape (batch_size,)
            q_values_pi_min = self.q_network.forward(obs, pi, True)
            policy_loss = torch.mean(log_prob * self.alpha - q_values_pi_min)

        # alpha loss
        if self.log_alpha_tensor is not None:
            alpha_loss = -(self.log_alpha_tensor * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha_tensor.exp().item()

        self.q_optimizer.zero_grad()
        q_values_loss.backward()
        self.q_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def get_alpha(self):
        return self.alpha

    @torch.no_grad()
    def predict_batch(self, state):
        obs = torch.from_numpy(state).type(FloatTensor)
        action_distribution = self.policy_net.forward_action(obs)
        actions = action_distribution.sample().cpu().numpy()
        return actions

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.policy_net.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(state_dict)

    def train(self, env, exp_name, num_epochs, epoch_length, prefill_steps,
              replay_pool_size, batch_size, logdir=None, checkpoint_path=None,
              **kwargs):

        logger = EpochLogger(output_dir=logdir, exp_name=exp_name)
        sampler = StepSampler(prefill_steps=prefill_steps, logger=logger)

        replay_pool = TransitionReplayBuffer(
            capacity=replay_pool_size,
            obs_shape=env.single_observation_space.shape,
            obs_dtype=env.single_observation_space.dtype,
            ac_shape=env.single_action_space.shape,
            ac_dtype=env.single_action_space.dtype,
        )

        sampler.initialize(env, self, replay_pool)

        if checkpoint_path is None:
            dummy_env = env.env_fns[0]()
            checkpoint_path = os.path.join(logger.get_output_dir(), dummy_env.spec.id)
            del dummy_env

        best_mean_episode_reward = -np.inf
        timer = Timer()
        total_timesteps = prefill_steps // env.num_envs * prefill_steps

        timer.reset()
        for epoch in range(num_epochs):
            for _ in tqdm(range(epoch_length), desc='Epoch {}/{}'.format(epoch + 1, num_epochs)):
                sampler.sample()
                obs, actions, next_obs, reward, done = replay_pool.sample(batch_size)
                self.update(obs=obs, actions=actions, next_obs=next_obs, done=done, reward=reward)
                self.update_target()

            total_timesteps += epoch_length * env.num_envs
            # save best model
            avg_return = logger.get_stats('EpReward')[0]

            if avg_return > best_mean_episode_reward:
                best_mean_episode_reward = avg_return
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)

            # logging
            logger.log_tabular('Time Elapsed', timer.get_time_elapsed())
            logger.log_tabular('EpReward', with_min_and_max=True)
            logger.log_tabular('EpLength', average_only=True, with_min_and_max=True)
            logger.log_tabular('TotalSteps', total_timesteps)
            logger.log_tabular('TotalEpisodes', sampler.get_total_episode())
            logger.log_tabular('BestAvgReward', best_mean_episode_reward)
            logger.log_tabular('Alpha', self.get_alpha())
            logger.log_tabular('Replay Size', len(replay_pool))
            logger.dump_tabular()
