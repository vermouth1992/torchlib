import os

import numpy as np
import tensorflow as tf
from tensorlib import drl
from tensorlib.drl.models.policy import BaseStochasticPolicyValue
from tensorlib.utils.logx import EpochLogger
from tensorlib.utils.timer import Timer

from .utils import PPOReplayBuffer, PPOSampler


class Agent(drl.BaseAgent):
    def __init__(self, policy_net: BaseStochasticPolicyValue, learning_rate=1e-3, lam=1., clip_param=0.2,
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
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_grad_norm)
        self.baseline_loss = tf.keras.losses.MSE
        self.lam = lam
        self.max_grad_norm = max_grad_norm

        self.target_kl = target_kl
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def predict_batch(self, states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        return self.policy_net.select_action(states).numpy()

    def predict_log_prob_batch(self, state, action):
        data_loader = tf.data.Dataset.from_tensor_slices((state, action)).batch(64)
        log_probs = []
        for state, action in data_loader:
            log_probs.append(self.policy_net.predict_log_prob(state, action))
        log_probs = tf.concat(log_probs, axis=0).numpy()
        return log_probs

    def predict_state_value_batch(self, state):
        """ compute the state value using nn baseline

        Args:
            state: (batch_size, ob_dim)

        Returns: (batch_size,)

        """
        data_loader = tf.data.Dataset.from_tensor_slices(state).batch(64)
        values = []
        for obs in data_loader:
            values.append(self.policy_net.predict_value(obs))
        values = tf.concat(values, axis=0).numpy()
        return values

    @tf.function
    def _update_policy_step(self, observation, action, discount_rewards, advantage, old_log_prob):
        print('Creating ppo policy step update graph with batch size {}'.format(observation.shape[0]))
        with tf.GradientTape() as tape:
            distribution, raw_baselines = self.policy_net(observation, training=True)
            entropy_loss = tf.reduce_mean(distribution.entropy())
            log_prob = distribution.log_prob(action)
            negative_approx_kl = log_prob - old_log_prob
            negative_approx_kl_mean = tf.reduce_mean(-negative_approx_kl)

            ratio = tf.exp(negative_approx_kl)
            surr1 = ratio * advantage
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            value_loss = self.baseline_loss(raw_baselines, discount_rewards)

            loss = policy_loss - entropy_loss * self.entropy_coef + value_loss * self.value_coef

        if negative_approx_kl_mean <= 1.5 * self.target_kl:
            # only update within trust region.
            gradients = tape.gradient(loss, self.policy_net.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

        return policy_loss, value_loss, entropy_loss, negative_approx_kl_mean

    def update_policy(self, data_loader, epoch, logger):
        for _ in tf.range(epoch):
            for batch_sample in data_loader:
                observation, action, discount_rewards, advantage, old_log_prob = batch_sample
                policy_loss, value_loss, entropy_loss, negative_approx_kl_mean = \
                    self._update_policy_step(observation, action, discount_rewards, advantage, old_log_prob)

                logger.store(PolicyLoss=policy_loss.numpy())
                logger.store(ValueLoss=value_loss.numpy())
                logger.store(EntropyLoss=entropy_loss.numpy())
                logger.store(NegativeAvgKL=negative_approx_kl_mean.numpy())

    @property
    def state_dict(self):
        return self.policy_net.trainable_weights

    def load_state_dict(self, states):
        self.policy_net.load_state_dict(states)

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        self.policy_net.save_weights(checkpoint_path, save_format='h5')

    def load_checkpoint(self, checkpoint_path):
        print('Load checkpoint from {}'.format(checkpoint_path))
        self.policy_net.load_weights(checkpoint_path)

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
