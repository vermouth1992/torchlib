"""
Pytorch implementation of proximal policy optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torchlib.deep_rl.policy_gradient.vanilla as vanilla_pg
from torch.utils.data import TensorDataset, DataLoader
from torchlib.common import eps, enable_cuda

from .utils import compute_gae, compute_sum_of_rewards


class Agent(vanilla_pg.Agent):
    def __init__(self, policy_net: nn.Module, policy_optimizer, discrete, nn_baseline,
                 nn_baseline_optimizer, lam=1., clip_param=0.2, entropy_coef=0.01):
        super(Agent, self).__init__(policy_net, policy_optimizer, discrete, nn_baseline,
                                    nn_baseline_optimizer, lam)
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef

    def construct_dataset(self, paths, gamma):
        rewards = compute_sum_of_rewards(paths, gamma)
        observation = np.concatenate([path["observation"] for path in paths])
        if self.nn_baseline:
            advantage = compute_gae(paths, gamma, self.nn_baseline, self.lam, np.mean(rewards), np.std(rewards))
        else:
            advantage = rewards

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

        with torch.no_grad():
            if enable_cuda:
                old_distribution = self.get_action_distribution(observation.cuda())
                old_log_prob = old_distribution.log_prob(actions.cuda()).cpu()
            else:
                old_distribution = self.get_action_distribution(observation)
                old_log_prob = old_distribution.log_prob(actions).cpu()

        return actions, advantage, observation, rewards, old_log_prob

    def update_policy(self, dataset, epoch=4):
        # construct a dataset using paths containing (action, observation, old_log_prob)
        actions, advantage, observation, rewards, old_log_prob = dataset
        torch_dataset = TensorDataset(observation, actions, old_log_prob, rewards, advantage)

        kwargs = {'num_workers': 1, 'pin_memory': True} if enable_cuda else {}

        # note that drop last is important because if there is just 1 reward, std will result in nan.
        data_loader = DataLoader(torch_dataset, batch_size=32, drop_last=True, shuffle=True, **kwargs)

        for _ in range(epoch):
            for batch_sample in data_loader:
                observation, action, old_log_prob, discount_rewards, advantage = batch_sample
                if enable_cuda:
                    observation = observation.cuda()
                    action = action.cuda()
                    old_log_prob = old_log_prob.cuda()
                    discount_rewards = discount_rewards.cuda()
                    advantage = advantage.cuda()

                self.policy_optimizer.zero_grad()
                # update policy
                distribution = self.get_action_distribution(observation)
                entropy_loss = distribution.entropy().mean()

                log_prob = distribution.log_prob(action)

                assert log_prob.shape == advantage.shape, 'log_prob length {}, advantage length {}'.format(
                    log_prob.shape,
                    advantage.shape)

                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - entropy_loss * self.entropy_coef
                loss.backward()
                self.policy_optimizer.step()

                # update baseline
                self.update_baseline(observation, discount_rewards)


train = vanilla_pg.train
