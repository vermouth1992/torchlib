"""
Pytorch implementation of proximal policy optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torchlib.deep_rl.policy_gradient.vanilla as vanilla_pg
from torchlib.common import eps, enable_cuda
from torchlib.dataset.utils import create_data_loader

from .utils import compute_gae, compute_sum_of_rewards


class Agent(vanilla_pg.Agent):
    def __init__(self, policy_net: nn.Module, policy_optimizer, discrete, init_hidden_unit, lam=1., clip_param=0.2,
                 entropy_coef=0.01, value_coef=0.5):
        super(Agent, self).__init__(policy_net, policy_optimizer, discrete, init_hidden_unit, True, lam, value_coef)
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef

    def construct_dataset(self, paths, gamma):
        rewards = compute_sum_of_rewards(paths, gamma)
        observation = np.concatenate([path["observation"] for path in paths])
        hidden = np.concatenate([path["hidden"] for path in paths])
        advantage = compute_gae(paths, gamma, self.policy_net, self.lam, np.mean(rewards), np.std(rewards))

        # reshape all episodes to a single large batch
        actions = []
        for path in paths:
            actions.extend(path['actions'])

        # normalize advantage
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + eps)
        actions = torch.Tensor(actions)
        advantage = torch.Tensor(advantage)
        rewards = torch.Tensor(rewards)
        observation = torch.Tensor(observation)
        hidden = torch.Tensor(hidden)

        with torch.no_grad():
            if enable_cuda:
                old_distribution, _ = self.get_action_distribution(observation.cuda(), hidden.cuda())
                old_log_prob = old_distribution.log_prob(actions.cuda()).cpu()
            else:
                old_distribution, _ = self.get_action_distribution(observation, hidden)
                old_log_prob = old_distribution.log_prob(actions).cpu()

        return actions, advantage, observation, rewards, old_log_prob, hidden

    def update_policy(self, dataset, epoch=4):
        # construct a dataset using paths containing (action, observation, old_log_prob)
        data_loader = create_data_loader(dataset, batch_size=32, shuffle=True, drop_last=True)

        for _ in range(epoch):
            for batch_sample in data_loader:
                action, advantage, observation, discount_rewards, old_log_prob, hidden = batch_sample
                if enable_cuda:
                    observation = observation.cuda()
                    action = action.cuda()
                    old_log_prob = old_log_prob.cuda()
                    discount_rewards = discount_rewards.cuda()
                    advantage = advantage.cuda()
                    hidden = hidden.cuda()

                self.policy_optimizer.zero_grad()
                # update policy
                distribution, _ = self.get_action_distribution(observation, hidden)

                entropy_loss = distribution.entropy().mean()

                log_prob = distribution.log_prob(action)

                assert log_prob.shape == advantage.shape, 'log_prob length {}, advantage length {}'.format(
                    log_prob.shape,
                    advantage.shape)

                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.get_baseline_loss(observation, hidden, discount_rewards)

                loss = policy_loss - entropy_loss * self.entropy_coef + self.value_coef * value_loss
                loss.backward()
                self.policy_optimizer.step()


train = vanilla_pg.train
