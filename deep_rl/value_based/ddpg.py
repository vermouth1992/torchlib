import copy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchlib.common import FloatTensor, enable_cuda, eps
from torchlib.deep_rl import BaseAgent
from torchlib.deep_rl.envs.wrappers import get_wrapper_by_name
from torchlib.deep_rl.utils import ReplayBuffer, PrioritizedReplayBuffer, ReplayBufferFrame, LinearSchedule
from torchlib.utils.random import set_global_seeds


class ActorNetwork(BaseAgent):
    def __init__(self, actor, optimizer, tau):
        self.tau = tau
        self.optimizer = optimizer

        self.actor_network = actor
        self.target_actor_network = copy.deepcopy(actor)
        if enable_cuda:
            self.actor_network.cuda()
            self.target_actor_network.cuda()

    def train(self, inputs, a_gradient):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        a_gradient = torch.from_numpy(a_gradient).type(FloatTensor)
        self.optimizer.zero_grad()
        actions = self.actor_network(inputs)
        actions.backward(-a_gradient)
        self.optimizer.step()

    def predict(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        return self.actor_network(inputs).data.cpu().numpy()

    def predict_target(self, inputs):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        return self.target_actor_network(inputs).data.cpu().numpy()

    def update_target_network(self):
        source = self.actor_network
        target = self.target_actor_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.actor_network.state_dict(), checkpoint_path)
        print('Save ddpg actor checkpoint to {}'.format(checkpoint_path))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.actor_network.load_state_dict(state_dict)
        print('Load ddpg actor checkpoint from {}'.format(checkpoint_path))


class CriticNetwork(object):
    def __init__(self, critic, optimizer, tau):
        self.optimizer = optimizer
        self.tau = tau
        self.loss = nn.SmoothL1Loss(reduction='none')

        self.critic_network = critic
        self.target_critic_network = copy.deepcopy(critic)

        if enable_cuda:
            self.critic_network.cuda()
            self.target_critic_network.cuda()
            self.loss.cuda()

    def train(self, inputs, action, predicted_q_value, importance_weights):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        predicted_q_value = torch.from_numpy(predicted_q_value).type(FloatTensor)
        importance_weights = torch.from_numpy(importance_weights).type(FloatTensor)

        self.optimizer.zero_grad()
        q_value = self.critic_network(inputs, action)
        delta = (predicted_q_value - q_value).data.cpu().numpy()

        loss = self.loss(q_value, predicted_q_value)
        loss = (loss * importance_weights).mean()
        loss.backward()
        self.optimizer.step()
        return q_value.data.cpu().numpy(), delta

    def predict(self, inputs, action):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        return self.critic_network(inputs, action).data.cpu().numpy()

    def predict_target(self, inputs, action):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        return self.target_critic_network(inputs, action).data.cpu().numpy()

    def action_gradients(self, inputs, actions):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        actions = Variable(torch.from_numpy(actions).type(FloatTensor), requires_grad=True)
        q_value = self.critic_network(inputs, actions)
        q_value = torch.mean(q_value)
        return torch.autograd.grad(q_value, actions)[0].data.cpu().numpy(), None

    def update_target_network(self):
        source = self.critic_network
        target = self.target_critic_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.critic_network.state_dict(), checkpoint_path)
        print('Save ddpg critic checkpoint to {}'.format(checkpoint_path))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.critic_network.load_state_dict(state_dict)
        print('Load ddpg critic checkpoint from {}'.format(checkpoint_path))


def train(env, actor, critic, actor_noise, total_timesteps, replay_buffer_type='normal', replay_buffer_config=None,
          batch_size=64, gamma=0.99, learn_starts=64, learning_freq=4, seed=1996, log_every_n_steps=10000,
          actor_checkpoint_path=None, critic_checkpoint_path=None):
    actor.update_target_network()
    critic.update_target_network()

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

        if actor_noise == 'param_noise':
            raise NotImplementedError
        else:
            if replay_buffer_type == 'frame':
                action = actor.predict(
                    np.expand_dims(replay_buffer.encode_recent_observation(), axis=0))[0]
            else:
                action = actor.predict(np.expand_dims(previous_observation, axis=0))[0]

            action += actor_noise()

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
                weights = np.ones_like(r_batch)
            elif replay_buffer_type == 'prioritized':
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(global_step))
                (s_batch, a_batch, r_batch, s2_batch, t_batch, weights, batch_idxes) = experience
            else:
                raise NotImplementedError

            target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

            y_i = np.expand_dims(r_batch, axis=-1) + gamma * target_q * (1 - np.expand_dims(t_batch, axis=-1))

            # Update the critic given the targets
            predicted_q_value, delta = critic.train(s_batch, a_batch, y_i, weights)

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(s_batch)
            grads = critic.action_gradients(s_batch, a_outs)
            actor.train(s_batch, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            if replay_buffer_type == 'prioritized':
                new_priorities = np.abs(delta) + eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if global_step % log_every_n_steps == 0:
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            last_one_hundred_episode_reward = episode_rewards[-100:]
            mean_episode_reward = np.mean(last_one_hundred_episode_reward)
            print('------------')
            if mean_episode_reward > best_mean_episode_reward:
                if actor_checkpoint_path:
                    actor.save_checkpoint(actor_checkpoint_path)
                if critic_checkpoint_path:
                    critic.save_checkpoint(critic_checkpoint_path)

            std_episode_reward = np.std(last_one_hundred_episode_reward)
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            print("Timestep {}/{}".format(global_step, total_timesteps))
            print("mean reward (100 episodes) {:.2f}. std {:.2f}".format(mean_episode_reward, std_episode_reward))
            print('reward range [{:.2f}, {:.2f}]'.format(np.min(last_one_hundred_episode_reward),
                                                         np.max(last_one_hundred_episode_reward)))
            print("best mean reward {:.2f}".format(best_mean_episode_reward))
            print("episodes %d" % len(episode_rewards))
