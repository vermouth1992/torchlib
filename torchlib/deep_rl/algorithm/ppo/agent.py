import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchlib import deep_rl
from torchlib.common import move_tensor_to_gpu, enable_cuda, convert_to_tensor
from torchlib.dataset.utils import create_data_loader
from torchlib.utils.logx import EpochLogger
from torchlib.utils.timer import Timer

from .utils import PPOReplayBuffer, PPOSampler


class Agent(deep_rl.BaseAgent):
    def __init__(self, policy_net: nn.Module, learning_rate=1e-3, lam=1., clip_param=0.2,
                 entropy_coef=0.01, value_coef=0.5, target_kl=0.05, max_grad_norm=0.5, **kwargs):
        """

        Args:
            policy_net: The policy net must implement following methods:
                - forward: takes obs and return action_distribution and value
                - forward_action: takes obs and return action_distribution
                - forward_value: takes obs and return value.
            The advantage is that we can save computation if we only need to fetch parts of the graph. Also, we can
            implement policy and value in both shared and non-shared way.
            learning_rate:
            lam:
            clip_param:
            entropy_coef:
            target_kl:
            max_grad_norm:
        """
        self.policy_net = policy_net
        if enable_cuda:
            self.policy_net.cuda()
        self.policy_optimizer = Adam(policy_net.parameters(), lr=learning_rate)
        self.baseline_loss = nn.MSELoss()
        self.lam = lam
        self.max_grad_norm = max_grad_norm

        self.target_kl = target_kl
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    @torch.no_grad()
    def predict_batch(self, states):
        states = convert_to_tensor(states.astype(np.float32))
        action_distribution = self.policy_net.forward_action(states)
        return action_distribution.sample().cpu().numpy()

    @torch.no_grad()
    def predict_log_prob_batch(self, state, action):
        data_loader = create_data_loader((state, action), batch_size=32, shuffle=False, drop_last=False)
        log_probs = []
        for obs, action in data_loader:
            obs = move_tensor_to_gpu(obs)
            action = move_tensor_to_gpu(action)
            action_distribution = self.policy_net.forward_action(obs)
            log_probs.append(action_distribution.log_prob(action))
        log_probs = torch.cat(log_probs, dim=0).cpu().numpy()
        return log_probs

    @torch.no_grad()
    def predict_state_value_batch(self, state):
        """ compute the state value using nn baseline

        Args:
            state: (batch_size, ob_dim)

        Returns: (batch_size,)

        """
        data_loader = create_data_loader((state,), batch_size=32, shuffle=False, drop_last=False)
        values = []
        for obs in data_loader:
            obs = move_tensor_to_gpu(obs[0])
            values.append(self.policy_net.forward_value(obs))
        values = torch.cat(values, dim=0).cpu().numpy()
        return values

    def update_policy(self, data_loader, epoch, logger):
        for epoch_index in range(epoch):
            for batch_sample in data_loader:

                with torch.autograd.detect_anomaly():

                    observation, action, discount_rewards, advantage, old_log_prob = move_tensor_to_gpu(batch_sample)
                    self.policy_optimizer.zero_grad()
                    # update policy
                    distribution, raw_baselines = self.policy_net.forward(observation)
                    entropy_loss = distribution.entropy().mean()
                    log_prob = distribution.log_prob(action)

                    assert log_prob.shape == advantage.shape, 'log_prob length {}, advantage length {}'.format(
                        log_prob.shape,
                        advantage.shape)

                    # if approximated kl is larger than 1.5 target_kl, we early stop training of this batch
                    negative_approx_kl = log_prob - old_log_prob
                    negative_approx_kl_mean = torch.mean(-negative_approx_kl).item()

                    if negative_approx_kl_mean > 1.5 * self.target_kl:
                        # print('Early stopping this iteration. Current kl {:.4f}. Current epoch index {}'.format(
                        #     negative_approx_kl_mean, epoch_index))
                        continue

                    ratio = torch.exp(negative_approx_kl)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = self.baseline_loss(raw_baselines, discount_rewards)

                    loss = policy_loss - entropy_loss * self.entropy_coef + value_loss * self.value_coef

                    nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

                    loss.backward()
                    self.policy_optimizer.step()

                logger.store(PolicyLoss=policy_loss.item())
                logger.store(ValueLoss=value_loss.item())
                logger.store(EntropyLoss=entropy_loss.item())
                logger.store(NegativeAvgKL=negative_approx_kl_mean)

    @property
    def state_dict(self):
        return self.policy_net.state_dict()

    def load_state_dict(self, states):
        self.policy_net.load_state_dict(states)

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        print('Load checkpoint from {}'.format(checkpoint_path))
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def train(self, env, exp_name, num_epoch, num_updates, gamma, min_steps_per_batch, batch_size=128, alpha=0.9,
              logdir=None, checkpoint_path=None, **kwargs):
        # create logger
        config = locals()
        config['self'] = str(self.policy_net)
        del config['env']
        logger = EpochLogger(output_dir=logdir, exp_name=exp_name)

        logger.save_config(config)
        if checkpoint_path is None:
            dummy_env = env.env_fns[0]()
            checkpoint_path = os.path.join(logger.get_output_dir(), dummy_env.spec.id)
            del dummy_env

        # create sampler and pool
        sampler = PPOSampler(min_steps_per_batch=min_steps_per_batch, logger=logger)
        replay_buffer = PPOReplayBuffer(gamma=gamma, lam=self.lam, policy=self, alpha=alpha)
        sampler.initialize(env=env, policy=self, pool=replay_buffer)

        # initialize training progress
        total_timesteps = 0
        best_avg_return = -np.inf
        timer = Timer()
        timer.reset()
        for itr in range(num_epoch):
            replay_buffer.clear()
            sampler.sample_trajectories()

            # calculate statistics
            total_timesteps += len(replay_buffer)
            currentEpReward = logger.get_stats('EpReward')[0]
            if currentEpReward >= best_avg_return:
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path=checkpoint_path + '_best.ckpt')
                best_avg_return = currentEpReward

            train_data_loader = replay_buffer.random_iterator(batch_size)
            self.update_policy(train_data_loader, epoch=num_updates, logger=logger)

            logger.log_tabular('Epoch (Total {})'.format(num_epoch), itr + 1)
            logger.log_tabular('Time Elapsed', timer.get_time_elapsed())
            logger.log_tabular('BestEpReward', best_avg_return)
            logger.log_tabular('EpReward', with_min_and_max=True)
            logger.log_tabular('EpLength', average_only=True, with_min_and_max=True)
            logger.log_tabular('TotalSteps', total_timesteps)
            logger.log_tabular('ValueRunningMean', replay_buffer.running_value_mean)
            logger.log_tabular('ValueRunningStd', replay_buffer.running_value_std)
            logger.log_tabular('PolicyLoss', average_only=True)
            logger.log_tabular('ValueLoss', average_only=True)
            logger.log_tabular('EntropyLoss', average_only=True)
            logger.log_tabular('NegativeAvgKL', average_only=True)
            logger.dump_tabular()
