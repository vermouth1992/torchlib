"""
BiGAN/ALI adds another encoder that encoder the input to latent space z.
The discriminator mark real/fake using the joint distribution of (x, z).
We don't provide the label to the network and hope that it can automatically
captures unsupervised features that can be directly used for downstream tasks.
"""

import numpy as np
import torch
from torch.autograd import Variable

from ....common import enable_cuda, FloatTensor


class ALI(object):
    def __init__(self, generator, encoder, discriminator, optimizer_G, optimizer_D, code_size, validity_loss_f):
        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.code_size = code_size
        self.validity_loss_f = validity_loss_f

        if enable_cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.encoder.cuda()
            self.validity_loss_f.cuda()

    def _set_to_train(self):
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()

    def reconstruct(self, data):
        self._set_to_eval()
        z = self.encoder(data)
        recon = self.generator(z)
        self._set_to_train()
        return recon

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

    def inference(self, data):
        self._set_to_eval()
        z = self.encoder(data)
        self._set_to_train()
        return z

    def _train(self, real_data, z, valid, fake):
        # re-parameter trick
        real_z = self.encoder(real_data)
        fake_data = self.generator(z)

        # accumulate discriminator gradient
        self.optimizer_D.zero_grad()
        validity_real = self.discriminator(real_data, real_z)
        d_real_loss = self.validity_loss_f(validity_real, valid)
        validity_fake = self.discriminator(fake_data, z)
        d_fake_loss = self.validity_loss_f(validity_fake, fake)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        g_real_loss = self.validity_loss_f(validity_real, fake)
        g_fake_loss = self.validity_loss_f(validity_fake, valid)
        g_loss = g_real_loss + g_fake_loss
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss, g_loss, validity_real.data.mean(), validity_fake.data.mean()

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        state = {
            'net_d': self.discriminator.state_dict(),
            'net_g': self.generator.state_dict(),
            'net_e': self.encoder.state_dict(),
            'optimizer_d': self.optimizer_D.state_dict(),
            'optimizer_g': self.optimizer_G.state_dict()
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        if enable_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.discriminator.load_state_dict(checkpoint['net_d'])
        self.generator.load_state_dict(checkpoint['net_g'])
        self.encoder.load_state_dict(checkpoint['net_e'])
        if all:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_g'])
