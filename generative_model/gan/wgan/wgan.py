"""
PyTorch implementation of Wasserstein GAN. https://arxiv.org/abs/1701.07875.
WGAN differs from vanilla GAN in the sense of distance measurement.
Vanilla GAN uses JS divergence, which may not converge.
WGAN uses Earth-Mover (EM) distance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torchlib.common import FloatTensor, enable_cuda
from torchlib.utils.torch_layer_utils import change_model_trainable


class WassersteinGAN(object):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, optimizer_G: optim.Optimizer,
                 optimizer_D: optim.Optimizer, code_size: int):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.code_size = code_size

        if enable_cuda:
            self.generator.cuda()
            self.discriminator.cuda()

    def _set_to_train(self):
        self.generator.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def _train_dis(self, real_data, z, c=0.01):
        change_model_trainable(self.generator, False)
        self.optimizer_D.zero_grad()
        fake_data = self.generator(z)
        validity_real = self.discriminator(real_data)
        validity_fake = self.discriminator(fake_data)
        dis_loss = -(torch.mean(validity_real) - torch.mean(validity_fake))
        dis_loss.backward()
        self.optimizer_D.step()

        # Weight clipping
        for p in self.discriminator.parameters():
            p.data.clamp_(-c, c)

        change_model_trainable(self.generator, True)

        return dis_loss, validity_real.data.mean(), validity_fake.data.mean()

    def _train_gen(self, z):
        change_model_trainable(self.discriminator, False)
        self.optimizer_G.zero_grad()
        validity_fake = self.discriminator(self.generator(z))
        gen_loss = -torch.mean(validity_fake)
        gen_loss.backward()
        self.optimizer_G.step()
        change_model_trainable(self.discriminator, True)
        return gen_loss, validity_fake.data.mean()

    def generate(self, mode, *args):
        self._set_to_eval()
        if mode == 'fixed':
            noise = torch.from_numpy(args[0]).type(FloatTensor)
        elif mode == 'num':
            num_samples = args[0]
            noise = Variable(FloatTensor(np.random.normal(0, 1, (num_samples, self.code_size))))
        else:
            raise ValueError('Unknown mode')

        gen_data = self.generator(noise)
        self._set_to_train()
        return gen_data

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        state = {
            'net_d': self.discriminator.state_dict(),
            'net_g': self.generator.state_dict(),
            'optimizer_d': self.optimizer_D.state_dict(),
            'optimizer_g': self.optimizer_G.state_dict()
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.discriminator.load_state_dict(checkpoint['net_d'])
        self.generator.load_state_dict(checkpoint['net_g'])
        if all:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_g'])
