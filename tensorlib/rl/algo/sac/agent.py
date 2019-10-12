import os
from typing import Tuple, Callable

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorlib import rl
from tensorlib.rl.utils.replay.replay import TransitionReplayBuffer
from tensorlib.rl.utils.replay.sampler import StepSampler
from tensorlib.utils.logx import EpochLogger
from tensorlib.utils.timer import Timer
from tensorlib.utils.weights import hard_update, soft_update
from tqdm.auto import tqdm

ds = tfp.distributions


class Agent(rl.BaseAgent):
    def __init__(self,
                 nets: Tuple[tf.keras.Model, Callable],
                 action_space,
                 learning_rate=3e-4,
                 no_automatic_alpha=False,
                 alpha=1.0,
                 tau=5e-3,
                 gamma=0.99,
                 **kwargs,
                 ):
        self.policy_net, q_network_maker = nets
        self.q_network = q_network_maker()
        self.target_q_network = q_network_maker()

        # self.target_q_network = tf.keras.models.clone_model(self.q_network)

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.automatic_alpha = not no_automatic_alpha

        hard_update(self.target_q_network, self.q_network)

        if self.automatic_alpha:
            if self.discrete:
                target_entropy = -np.log(action_space.n) * 0.95
            else:
                target_entropy = -np.prod(action_space.shape)
            self.log_alpha_tensor = tf.Variable(initial_value=0., dtype=tf.float32)
            self.alpha_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            self.target_entropy = target_entropy
            self.get_alpha = tf.function(func=lambda: tf.exp(self.log_alpha_tensor))
        else:
            self.get_alpha = lambda: alpha
        self.tau = tau
        self.gamma = gamma

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    @tf.function
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
        print('Building SAC update graph with batch size {}'.format(obs.shape[0]))

        next_action, next_action_log_prob = self.policy_net.predict_action_log_prob(next_obs)
        target_q_values = self.target_q_network.predict_min_value_with_action(next_obs, next_action) \
                          - self.get_alpha() * next_action_log_prob
        q_target = reward + self.gamma * (1.0 - done) * target_q_values

        # q loss
        with tf.GradientTape() as q_tape, tf.GradientTape() as policy_tape:
            q_values, q_values2 = self.q_network.predict_value_with_action(obs, actions)
            q_values_loss = tf.keras.losses.mse(q_values, q_target) + tf.keras.losses.mse(q_values2, q_target)
            # policy loss
            if self.discrete:
                # for discrete action space, we can directly compute kl divergence analytically without sampling
                action_distribution = self.policy_net.predict_action_distribution(obs)
                q_values_min = self.q_network.predict_min_value(obs)  # (batch_size, ac_dim)
                target_distribution = ds.Categorical(logits=q_values_min, dtype=tf.int64)
                policy_loss = tf.reduce_mean(ds.kl_divergence(action_distribution, target_distribution))
                log_prob = -action_distribution.entropy()
            else:
                action, log_prob = self.policy_net.predict_action_log_prob(obs)
                q_values_pi_min = self.q_network.predict_min_value_with_action(obs, action)
                policy_loss = tf.reduce_mean(log_prob * self.get_alpha() - q_values_pi_min)

            q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
            policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)

            self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))
            self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

            if self.automatic_alpha:
                with tf.GradientTape() as alpha_tape:
                    alpha_loss = -tf.reduce_mean(self.log_alpha_tensor * (log_prob + self.target_entropy))
                    alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha_tensor)
                    self.alpha_optimizer.apply_gradients(zip([alpha_gradient], [self.log_alpha_tensor]))

    def predict_batch(self, states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        return self.policy_net.select_action(states).numpy()

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        self.policy_net.save_weights(checkpoint_path, save_format='h5')

    def load_checkpoint(self, checkpoint_path):
        print('Load checkpoint from {}'.format(checkpoint_path))
        self.policy_net.load_weights(checkpoint_path)

    def train(self, env, exp_name, num_epochs, epoch_length, prefill_steps,
              replay_pool_size, batch_size, logdir=None, checkpoint_path=None,
              **kwargs):

        logger = EpochLogger(output_dir=logdir, exp_name=exp_name)

        if checkpoint_path is None:
            dummy_env = env.env_fns[0]()
            checkpoint_path = os.path.join(logger.get_output_dir(), dummy_env.spec.id)
            del dummy_env

        sampler = StepSampler(prefill_steps=prefill_steps, logger=logger)

        replay_pool = TransitionReplayBuffer(
            capacity=replay_pool_size,
            obs_shape=env.single_observation_space.shape,
            obs_dtype=env.single_observation_space.dtype,
            ac_shape=env.single_action_space.shape,
            ac_dtype=env.single_action_space.dtype,
        )

        sampler.initialize(env, self, replay_pool)

        best_mean_episode_reward = -np.inf
        timer = Timer()
        total_timesteps = prefill_steps // env.num_envs * prefill_steps

        timer.reset()
        for epoch in range(num_epochs):
            for _ in tqdm(range(epoch_length), desc='Epoch {}/{}'.format(epoch + 1, num_epochs)):
                sampler.sample()
                obs, actions, next_obs, reward, done = replay_pool.sample(batch_size)
                obs = tf.convert_to_tensor(obs)
                actions = tf.convert_to_tensor(actions)
                next_obs = tf.convert_to_tensor(next_obs)
                done = tf.cast(tf.convert_to_tensor(done), tf.float32)
                reward = tf.convert_to_tensor(reward)
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
