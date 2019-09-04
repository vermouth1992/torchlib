"""
Implement Twin-Delayed DDPG in Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018
The key difference with DDPG lies in
1. Add noise to target policy served as regularization to prevent overfitting to current best policy
2. Use clipped double Q function to avoid overestimation in Q value
3. Add Gaussian noise to explore at training time.
"""

import copy
import os

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import wrappers
from torchlib import deep_rl
from torchlib.common import convert_numpy_to_tensor, enable_cuda, FloatTensor
from torchlib.deep_rl.models import ActorModule, DoubleCriticModule
from torchlib.deep_rl.utils import ReplayBuffer
from torchlib.utils.random import set_global_seeds
from torchlib.utils.weight import soft_update, hard_update


class TD3(deep_rl.BaseAgent):
    def __init__(self,
                 actor_module, actor_optimizer,
                 critic_module, critic_optimizer,
                 ):
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

        self.target_actor_module = copy.deepcopy(self.actor_module)
        self.target_critic_module = copy.deepcopy(self.critic_module)
        hard_update(self.target_actor_module, self.actor_module)
        hard_update(self.target_critic_module, self.critic_module)

        if enable_cuda:
            self.actor_module.cuda()
            self.critic_module.cuda()
            self.target_actor_module.cuda()
            self.target_critic_module.cuda()

    @torch.no_grad()
    def predict(self, state):
        state = convert_numpy_to_tensor(np.expand_dims(state, axis=0))
        return self.actor_module.forward(state).cpu().numpy()[0]

    def state_dict(self):
        pass

    def load_state_dict(self, states):
        pass

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def update(self, replay_buffer, num_updates, policy_freq=2, batch_size=128, target_noise=0.2, clip_noise=0.5,
               tau=1e-3, gamma=0.99):
        for i in range(num_updates):
            transition = replay_buffer.sample(batch_size)
            s_batch, a_batch, r_batch, s2_batch, t_batch = convert_numpy_to_tensor(transition, location='gpu')

            r_batch = r_batch.type(FloatTensor)
            t_batch = t_batch.type(FloatTensor)

            # get ground truth q value
            with torch.no_grad():
                target_action_noise = torch.clamp(torch.randn_like(a_batch) * target_noise, min=-clip_noise,
                                                  max=clip_noise)
                target_action = torch.clamp(self.target_actor_module.forward(s2_batch) + target_action_noise,
                                            min=-1., max=1.)
                target_q = self.target_critic_module.forward(state=s2_batch, action=target_action, minimum=True)

                q_target = r_batch + gamma * target_q * (1 - t_batch)

            # critic loss
            q_values, q_values2 = self.critic_module.forward(s_batch, a_batch, minimum=False)
            q_values_loss = F.mse_loss(q_values, q_target) + F.mse_loss(q_values2, q_target)

            self.critic_optimizer.zero_grad()
            q_values_loss.backward()
            self.critic_optimizer.step()

            if i % policy_freq == 0:
                action = self.actor_module.forward(s_batch)
                q_values = self.critic_module.forward(s_batch, action, minimum=False)[0]
                loss = -torch.mean(q_values)
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

                soft_update(self.target_critic_module, self.critic_module, tau)
                soft_update(self.target_actor_module, self.actor_module, tau)

    def train(self, env, actor_noise=None, seed=1996,
              start_steps=10000, total_steps=1000000,
              replay_size=1000000, replay_buffer=None,
              num_updates=10, policy_freq=2, batch_size=128,
              target_noise=0.2, clip_noise=0.5, tau=5e-3, gamma=0.99,
              log_freq=1000):
        expt_dir = '/tmp/{}'.format(env.spec.id)
        env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

        set_global_seeds(seed)
        env.seed(seed)

        best_mean_episode_reward = -np.inf

        # create action noise for exploration
        if actor_noise is None:
            actor_noise = lambda: np.random.randn(*env.action_space.shape).astype(np.float32) * 0.1

        # create replay buffer
        if replay_buffer is None:
            replay_buffer = ReplayBuffer(size=replay_size)

        # warmup
        print('Warmup')
        previous_observation = env.reset()
        for _ in range(start_steps):
            action = env.action_space.sample().astype(np.float32)
            observation, reward, done, _ = env.step(action)
            replay_buffer.add(previous_observation, action, reward, observation, float(done))

            if done:
                previous_observation = env.reset()
            else:
                previous_observation = observation

        # actually training
        print('Start training')
        for global_step in range(total_steps):
            action = self.predict(previous_observation) + actor_noise()
            # perform clipping to ensure that it is within bound.
            action = np.clip(action, a_min=-1., a_max=1.)

            observation, reward, done, _ = env.step(action)

            replay_buffer.add(previous_observation, action, reward, observation, float(done))

            if done:
                previous_observation = env.reset()
            else:
                previous_observation = observation

            self.update(replay_buffer, num_updates, policy_freq, batch_size, target_noise, clip_noise, tau, gamma)

            if global_step % log_freq == 0:
                episode_rewards = deep_rl.envs.wrappers.get_wrapper_by_name(env, "Monitor").get_episode_rewards()
                last_one_hundred_episode_reward = episode_rewards[-100:]
                mean_episode_reward = np.mean(last_one_hundred_episode_reward)
                std_episode_reward = np.std(last_one_hundred_episode_reward)
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                print('------------')
                print("Timestep {}/{}".format(global_step, total_steps))
                print("mean reward (100 episodes) {:.2f}. std {:.2f}".format(mean_episode_reward, std_episode_reward))
                print('reward range [{:.2f}, {:.2f}]'.format(np.min(last_one_hundred_episode_reward),
                                                             np.max(last_one_hundred_episode_reward)))
                print("best mean reward {:.2f}".format(best_mean_episode_reward))
                print("episodes {}".format(len(episode_rewards)))


def make_default_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--nn_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--log_freq', type=int, default=1000)

    parser.add_argument('--max_episode_length', type=float, default=1000)

    parser.add_argument('--clip_noise', type=float, default=0.5)
    parser.add_argument('--target_noise', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--num_updates', type=int, default=4)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    return parser


def get_default_actor_critic(env: gym.Env, args):
    actor_module = ActorModule(size=args['nn_size'],
                               state_dim=env.observation_space.low.shape[0],
                               action_dim=env.action_space.low.shape[0])

    critic_module = DoubleCriticModule(size=args['nn_size'], state_dim=env.observation_space.low.shape[0],
                                       action_dim=env.action_space.low.shape[0])

    return actor_module, critic_module
