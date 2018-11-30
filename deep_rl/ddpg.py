import copy

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import MSELoss
from torchlib.common import FloatTensor, enable_cuda
from torchlib.utils.random.torch_random_utils import set_global_seeds

from .utils.replay.prioritized_experience_replay import rank_based
from .utils.replay.replay_buffer import ReplayBuffer


class ActorNetwork(object):
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

    def gather_state_dict(self):
        state_dict = {
            'network': self.actor_network.state_dict(),
            'target': self.target_actor_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def scatter_state_dict(self, state_dict, all=True):
        self.actor_network.load_state_dict(state_dict['network'])
        self.target_actor_network.load_state_dict(state_dict['target'])
        if all:
            self.optimizer.load_state_dict(state_dict['optimizer'])


class CriticNetwork(object):
    def __init__(self, critic, optimizer, tau):
        self.optimizer = optimizer
        self.tau = tau

        self.critic_network = critic
        self.target_critic_network = copy.deepcopy(critic)
        self.loss = MSELoss()

        if enable_cuda:
            self.critic_network.cuda()
            self.target_critic_network.cuda()
            self.loss.cuda()

    def train(self, inputs, action, predicted_q_value):
        inputs = torch.from_numpy(inputs).type(FloatTensor)
        action = torch.from_numpy(action).type(FloatTensor)
        predicted_q_value = torch.from_numpy(predicted_q_value).type(FloatTensor)

        self.optimizer.zero_grad()
        q_value = self.critic_network(inputs, action)
        output = self.loss(q_value, predicted_q_value)

        delta = (predicted_q_value - q_value).data.cpu().numpy()

        output.backward()
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

    def gather_state_dict(self):
        state_dict = {
            'network': self.critic_network.state_dict(),
            'target': self.target_critic_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        return state_dict

    def scatter_state_dict(self, state_dict, all=True):
        self.critic_network.load_state_dict(state_dict['network'])
        self.target_critic_network.load_state_dict(state_dict['target'])
        if all:
            self.optimizer.load_state_dict(state_dict['optimizer'])


class Trainer(object):
    def __init__(self, env, actor: ActorNetwork, critic: CriticNetwork, actor_noise, config, obs_normalizer=None,
                 action_processor=None):
        # config contains all the training parameters
        self.config = config

        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor

    def test(self, num_episode=100):
        reward_lst = []
        for episode in range(num_episode):
            current_reward = 0.
            observation = self.env.reset()
            for i in range(self.config['max step']):
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

        self.actor.update_target_network()
        self.critic.update_target_network()

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
            replay_buffer = rank_based.Experience(conf)
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
                action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                    axis=0) + self.actor_noise()

                if self.action_processor:
                    action_take = self.action_processor(action)
                else:
                    action_take = action
                # step forward
                observation, reward, done, _ = self.env.step(action_take)

                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation)

                # add to buffer
                replay_buffer.add(previous_observation, action, reward, done, observation)

                if replay_buffer.record_size >= learn_start:
                    # batch update
                    if not use_prioritized_buffer:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
                    else:
                        experience, w, rank_e_id = replay_buffer.sample_batch(global_step)
                        s_batch = np.array([_[0] for _ in experience])
                        a_batch = np.array([_[1] for _ in experience])
                        r_batch = np.array([_[2] for _ in experience])
                        t_batch = np.array([_[4] for _ in experience])
                        s2_batch = np.array([_[3] for _ in experience])
                    # Calculate targets
                    target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(batch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + gamma * target_q[k, 0])

                    # Update the critic given the targets
                    predicted_q_value, delta = self.critic.train(
                        s_batch, a_batch, np.reshape(y_i, (batch_size, 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    self.actor.train(s_batch, grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                    if use_prioritized_buffer:
                        replay_buffer.update_priority(rank_e_id, delta + np.expand_dims(w, axis=1))

                ep_reward += reward
                previous_observation = observation

                global_step += 1

                if done or j == self.config['max step'] - 1:
                    if use_prioritized_buffer:
                        replay_buffer.rebalance()
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j))))
                    break

            if save_every_episode > 0 and i % (save_every_episode + 1) == 0:
                self.save_checkpoint(checkpoint_path)

        self.save_checkpoint(checkpoint_path)
        print('Finish.')

    def predict(self, observation):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.

        Args:
            observation: observation provided by the environment

        Returns: action array with shape (batch_size, num_stocks + 1)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_checkpoint(self, checkpoint_path):
        state = {
            'actor': self.actor.gather_state_dict(),
            'critic': self.critic.gather_state_dict()
        }
        torch.save(state, checkpoint_path)
        print('Save checkpoint to {}'.format(checkpoint_path))

    def load_checkpoint(self, checkpoint_path, all=True):
        checkpoint = torch.load(checkpoint_path)
        self.actor.scatter_state_dict(checkpoint['actor'], all)
        self.critic.scatter_state_dict(checkpoint['critic'], all)
        print('Load checkpoint from {}'.format(checkpoint_path))