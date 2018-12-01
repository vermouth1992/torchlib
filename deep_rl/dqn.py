"""
Q network for vanilla/double/dueling DQN.
Q learning only works for discrete action space Discrete() in gym.
"""

import copy
from typing import Dict

import numpy as np
import torch
from torch.nn import SmoothL1Loss
from torchlib.common import FloatTensor, LongTensor, enable_cuda
from torchlib.utils.random.torch_random_utils import set_global_seeds

from .utils.replay.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class QNetwork(object):
    def __init__(self, network, optimizer, tau):
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.optimizer = optimizer
        self.tau = tau
        self.loss = SmoothL1Loss()

        if enable_cuda:
            self.network.cuda()
            self.target_network.cuda()
            self.loss.cuda()

    def train(self, inputs, actions, predicted_q_value):
        """ train the q network with one step

        Args:
            inputs: state of shape (batch_size, state_dim), or (batch, channel, img_h, img_w)
            actions: shape of (batch_size,)
            predicted_q_value: (batch_size, )

        Returns: q_value from network

        """
        actions = torch.from_numpy(actions).type(LongTensor)
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        predicted_q_value = torch.from_numpy(predicted_q_value).type(FloatTensor)
        self.optimizer.zero_grad()
        q_value = self.network(inputs).gather(1, actions.unsqueeze(1)).squeeze()
        output = self.loss(q_value, predicted_q_value)
        output.backward()
        self.optimizer.step()
        delta = (predicted_q_value - q_value).data.cpu().numpy()
        return q_value.data.cpu().numpy(), delta

    def predict_action(self, inputs):
        return np.argmax(self.compute_q_value(inputs), axis=1)

    def compute_q_value(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        q_value = self.network(inputs)
        return q_value.data.cpu().numpy()

    def compute_target_q_value(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        q_value = self.target_network(inputs)
        return q_value.data.cpu().numpy()

    def update_target_network(self):
        source = self.network
        target = self.target_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def gather_state_dict(self):
        state_dict = {
            'network': self.network.state_dict(),
            'target': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def scatter_state_dict(self, state_dict, all=True):
        self.network.load_state_dict(state_dict['network'])
        self.target_network.load_state_dict(state_dict['target'])
        if all:
            self.optimizer.load_state_dict(state_dict['optimizer'])


class Trainer(object):
    def __init__(self, env, q_network: QNetwork, exploration_criteria, config: Dict, obs_normalizer=None,
                 action_processor=None):
        """ Instantiate Trainer for DQN

        Args:
            env: gym env
            q_network: Q network
            exploration_criteria: a function takes predicted action and global step and produce action to take
            config: contains all the training parameters.
            obs_normalizer: normalize observation
            action_processor: process action
        """

        self.config = config

        self.env = env
        self.q_network = q_network
        self.exploration_criteria = exploration_criteria
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor

    def test(self, num_episode=100, render=False):
        reward_lst = []
        for episode in range(num_episode):
            current_reward = 0.
            observation = self.env.reset()
            for i in range(self.config['max step']):
                if render:
                    self.env.render()
                action = self.predict_single(observation)
                observation, reward, done, info = self.env.step(action)
                current_reward += reward
                if done:
                    break
            print("Episode: {}, Reward: {}".format(episode, current_reward))
            reward_lst.append(current_reward)
        reward_lst = np.array(reward_lst)
        average = reward_lst.mean()
        std = reward_lst.std()
        print('Performance: {:.2f}±{:.2f}'.format(average, std))

    def train(self, num_episode, checkpoint_path, save_every_episode=1, verbose=True, debug=False):
        self.q_network.update_target_network()
        set_global_seeds(self.config['seed'])
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        use_prioritized_buffer = self.config['use_prioritized_buffer']
        if use_prioritized_buffer:
            # Initialize replay memory
            conf = {'size': self.config['buffer size'],
                    'batch_size': batch_size,
                    'learn_start': 1000
                    }
            replay_buffer = PrioritizedReplayBuffer(self.config['buffer size'], alpha=0.6)
            learn_start = conf['learn_start']
        else:
            replay_buffer = ReplayBuffer(self.config['buffer size'])
            learn_start = batch_size

        global_step = 0
        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(replay_buffer.size()))

            previous_observation = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)
            ep_reward = 0
            ep_ave_max_q = 0
            # keeps sampling until done
            for j in range(self.config['max step']):
                if np.random.rand() < self.exploration_criteria(global_step):
                    action = self.env.action_space.sample()
                else:
                    action = self.q_network.predict_action(np.expand_dims(previous_observation, axis=0))[0]

                if self.action_processor:
                    action_take = self.action_processor(action)
                else:
                    action_take = action
                # step forward
                observation, reward, done, _ = self.env.step(action_take)

                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation)

                # add to buffer
                replay_buffer.add(previous_observation, action, reward, observation, float(done))

                if len(replay_buffer) >= learn_start:
                    # batch update
                    if not use_prioritized_buffer:
                        s_batch, a_batch, r_batch, s2_batch, t_batch = replay_buffer.sample(batch_size)
                    else:
                        experience, w, rank_e_id = replay_buffer.sample_batch(global_step)
                        s_batch = np.array([_[0] for _ in experience])
                        a_batch = np.array([_[1] for _ in experience])
                        r_batch = np.array([_[2] for _ in experience])
                        t_batch = np.array([_[4] for _ in experience])
                        s2_batch = np.array([_[3] for _ in experience])
                    # Calculate targets
                    target_q = self.q_network.compute_target_q_value(s2_batch)
                    if self.config['double_q']:
                        q_action = np.argmax(self.q_network.compute_q_value(s2_batch), axis=1)
                        target_q = target_q[np.arange(batch_size), q_action]
                    else:
                        target_q = np.max(target_q, axis=1)
                    y_i = r_batch + gamma * target_q * (1 - t_batch)

                    predicted_q_value, delta = self.q_network.train(s_batch, a_batch, y_i)
                    ep_ave_max_q += np.amax(predicted_q_value)

                    self.q_network.update_target_network()

                    if use_prioritized_buffer:
                        replay_buffer.update_priority(rank_e_id, np.expand_dims(w + delta, axis=1))

                ep_reward += reward
                previous_observation = observation

                global_step += 1

                if done or j == self.config['max step'] - 1:
                    if use_prioritized_buffer:
                        replay_buffer.rebalance()
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j))))
                    # print(self.exploration_criteria.epsilon)
                    break

            if save_every_episode > 0 and i % (save_every_episode + 1) == 0:
                self.save_checkpoint(checkpoint_path)

        self.save_checkpoint(checkpoint_path)
        print('Finish.')

    def predict_single(self, observation):
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.q_network.predict_action(np.expand_dims(observation, axis=0))[0]
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.q_network.gather_state_dict(), checkpoint_path)
        print('Save checkpoint to {}'.format(checkpoint_path))

    def load_checkpoint(self, checkpoint_path, all=True):
        checkpoint = torch.load(checkpoint_path)
        self.q_network.scatter_state_dict(checkpoint, all)
        print('Load checkpoint from {}'.format(checkpoint_path))
