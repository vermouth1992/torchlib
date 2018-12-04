"""
Q network for vanilla/double/dueling DQN.
Q learning only works for discrete action space Discrete() in gym.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss
from torchlib.common import FloatTensor, LongTensor, enable_cuda, eps
from torchlib.deep_rl.utils.atari_wrappers import get_wrapper_by_name
from torchlib.deep_rl.utils.replay.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, ReplayBufferFrame
from torchlib.deep_rl.utils.schedules import LinearSchedule, Schedule
from torchlib.utils.random.torch_random_utils import set_global_seeds


class QNetwork(object):
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer, optimizer_scheduler=None, tau=1e-3):
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.tau = tau
        self.loss = SmoothL1Loss(reduction='none')

        if enable_cuda:
            self.network.cuda()
            self.target_network.cuda()
            self.loss.cuda()

    def train(self, transition_batch, weights, gamma, double_q, grad_norm_clipping=10):
        """ train the q network with one step

        Args:
            inputs: state of shape (batch_size, state_dim), or (batch, channel, img_h, img_w)
            actions: shape of (batch_size,)
            predicted_q_value: (batch_size, )

        Returns: q_value from network

        """
        s_batch, a_batch, r_batch, s2_batch, t_batch = transition_batch
        s2_batch = torch.from_numpy(s2_batch).type(FloatTensor)
        t_batch = torch.from_numpy(t_batch).type(FloatTensor)
        r_batch = torch.from_numpy(r_batch).type(FloatTensor)
        target_q = self.compute_target_q_value(s2_batch)
        if double_q:
            q_action = torch.max(self.network.forward(s2_batch), dim=1)[1]
            target_q = target_q.gather(1, q_action.unsqueeze(1)).squeeze()
        else:
            target_q = torch.max(target_q, dim=1)[0]

        y_i = r_batch + gamma * target_q * (1 - t_batch)

        actions = torch.from_numpy(a_batch).type(LongTensor)
        inputs = torch.from_numpy(s_batch).type(FloatTensor)
        weights = torch.from_numpy(weights).type(FloatTensor)
        predicted_q_value = y_i.detach()
        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()
        self.optimizer.zero_grad()
        q_value = self.network(inputs).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss(q_value, predicted_q_value)
        loss = (loss * weights).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), grad_norm_clipping)
        self.optimizer.step()
        delta = (predicted_q_value - q_value).data.cpu().numpy()
        return q_value.data.cpu().numpy(), delta

    def predict_action(self, inputs):
        return np.argmax(self.compute_q_value(inputs), axis=1)

    def compute_q_value(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        with torch.no_grad():
            q_value = self.network(inputs)
        return q_value.data.cpu().numpy()

    def compute_target_q_value(self, inputs):
        q_value = self.target_network(inputs)
        return q_value

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.network.state_dict(), checkpoint_path)
        print('Save checkpoint to {}'.format(checkpoint_path))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.network.load_state_dict(state_dict)
        print('Load checkpoint from {}'.format(checkpoint_path))


def test(env, q_network, num_episode=100, frame_history_len=1, seed=1996):
    set_global_seeds(seed)
    env.seed(seed)
    reward_lst = []
    for i in range(num_episode):
        observation_lst = []
        done = False
        episode_reward = 0
        previous_observation = env.reset()
        observation_lst.append(previous_observation)
        for _ in range(frame_history_len - 1):
            action = env.action_space.sample()
            previous_observation, reward, done, _ = env.step(action)
            observation_lst.append(previous_observation)
            episode_reward += reward
        while not done:
            action = q_network.predict_action(np.expand_dims(np.concatenate(observation_lst, axis=-1), axis=0))[0]
            previous_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            observation_lst.pop(0)
            observation_lst.append(previous_observation)
        print('Episode: {}. Reward: {}'.format(i, episode_reward))
        reward_lst.append(episode_reward)
    print('Reward range [{}, {}]'.format(np.min(reward_lst), np.max(reward_lst)))
    print('Reward {}±{}'.format(np.mean(reward_lst), np.std(reward_lst)))


def train(env, q_network: QNetwork, exploration, total_timesteps, replay_buffer_type='normal',
          replay_buffer_config=None, batch_size=64, gamma=0.99, learn_starts=64, learning_freq=4, double_q=True,
          seed=1996, log_every_n_steps=10000, target_update_freq=3000, grad_norm_clipping=10, checkpoint_path=None):
    q_network.update_target_network()
    set_global_seeds(seed)
    env.seed(seed)
    if replay_buffer_type == 'normal':
        replay_buffer = ReplayBuffer(replay_buffer_config['size'])
    elif replay_buffer_type == 'prioritized':
        alpha = replay_buffer_config.get('alpha', 0.6)
        iter = replay_buffer_config.get('iter', total_timesteps)
        beta0 = replay_buffer_config.get('beta0', 0.4)
        replay_buffer = PrioritizedReplayBuffer(replay_buffer_config['size'], alpha=alpha)
        beta_schedule = LinearSchedule(iter, initial_p=beta0, final_p=1.0)
    elif replay_buffer_type == 'frame':
        frame_history_len = replay_buffer_config.get('frame_history_len', 4)
        replay_buffer = ReplayBufferFrame(replay_buffer_config['size'], frame_history_len)
    else:
        raise ValueError('Unknown replay buffer type.')

    previous_observation = env.reset()
    best_mean_episode_reward = -float('inf')

    num_updates = 0

    # warm the replay buffer
    print('Warm up replay buffer')
    for _ in range(learn_starts):
        if replay_buffer_type == 'frame':
            idx = replay_buffer.store_frame(previous_observation)

        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        if replay_buffer_type == 'frame':
            replay_buffer.store_effect(idx, action, reward, float(done))
        else:
            replay_buffer.add(previous_observation, action, reward, observation, float(done))

        if done:
            previous_observation = env.reset()
        else:
            previous_observation = observation

    print('Start training')
    for global_step in range(total_timesteps):
        if replay_buffer_type == 'frame':
            idx = replay_buffer.store_frame(previous_observation)

        if isinstance(exploration, Schedule):
            if np.random.rand() < exploration.value(global_step):
                action = env.action_space.sample()
            else:
                if replay_buffer_type == 'frame':
                    action = q_network.predict_action(
                        np.expand_dims(replay_buffer.encode_recent_observation(), axis=0))[0]
                else:
                    action = q_network.predict_action(np.expand_dims(previous_observation, axis=0))[0]

        elif exploration == 'param_noise':
            raise NotImplementedError

        else:
            raise ValueError('Unknown exploration')

        observation, reward, done, _ = env.step(action)

        if replay_buffer_type == 'frame':
            replay_buffer.store_effect(idx, action, reward, float(done))
        else:
            replay_buffer.add(previous_observation, action, reward, observation, float(done))

        if done:
            previous_observation = env.reset()
        else:
            previous_observation = observation

        if global_step % learning_freq == 0:
            if replay_buffer_type == 'normal' or replay_buffer_type == 'frame':
                s_batch, a_batch, r_batch, s2_batch, t_batch = replay_buffer.sample(batch_size)
                transition_batch = (s_batch, a_batch, r_batch, s2_batch, t_batch)
                weights = np.ones_like(r_batch)
            elif replay_buffer_type == 'prioritized':
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(global_step))
                (s_batch, a_batch, r_batch, s2_batch, t_batch, weights, batch_idxes) = experience
                transition_batch = (s_batch, a_batch, r_batch, s2_batch, t_batch)
            else:
                raise NotImplementedError

            predicted_q_value, delta = q_network.train(transition_batch, weights, gamma, double_q, grad_norm_clipping)

            if replay_buffer_type == 'prioritized':
                new_priorities = np.abs(delta) + eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            num_updates += 1

            if num_updates % target_update_freq == 0:
                q_network.update_target_network()

        if global_step % log_every_n_steps == 0:
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            last_one_hundred_episode_reward = episode_rewards[-100:]
            mean_episode_reward = np.mean(last_one_hundred_episode_reward)
            print('------------')
            if checkpoint_path and mean_episode_reward > best_mean_episode_reward:
                q_network.save_checkpoint(checkpoint_path)
            std_episode_reward = np.std(last_one_hundred_episode_reward)
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            print("Timestep {}/{}".format(global_step, total_timesteps))
            print("mean reward (100 episodes) {:.2f}. std {:.2f}".format(mean_episode_reward, std_episode_reward))
            print('reward range [{:.2f}, {:.2f}]'.format(np.min(last_one_hundred_episode_reward),
                                                         np.max(last_one_hundred_episode_reward)))
            print("best mean reward {:.2f}".format(best_mean_episode_reward))
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(global_step))
