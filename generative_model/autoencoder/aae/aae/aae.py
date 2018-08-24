"""
Adversarial auto encoder in Pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .....common import enable_cuda, FloatTensor
from .....utils.random.sampler import BaseSampler


class AAE(object):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 optimizer_G: optim.Optimizer, optimizer_D: optim.Optimizer,
                 code_size: int, validity_loss_f, recon_loss_f, latent_sampler: BaseSampler,
                 alpha=0.5):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.code_size = code_size
        self.validity_loss_f = validity_loss_f
        self.recon_loss_f = recon_loss_f
        self.latent_sampler = latent_sampler
        self.alpha = alpha

        if enable_cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.validity_loss_f.cuda()
            self.recon_loss_f.cuda()

    def _train_step(self, images, valid, fake):
        self.optimizer_G.zero_grad()
        # forward pass
        z = self.encoder(images)
        recon_images = self.decoder(z)
        recon_loss = self.recon_loss_f(recon_images, images)

        out = self.discriminator(z)
        validity_g_fake = out
        adversarial_loss = self.validity_loss_f(validity_g_fake, valid)

        g_loss = self.alpha * adversarial_loss + (1 - self.alpha) * recon_loss
        g_loss.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        real_data = self.sample_latent_code(valid.shape[0])
        out = self.discriminator(real_data)
        validity_real = out
        d_real_loss = self.validity_loss_f(validity_real, valid)

        out = self.discriminator(z.detach())
        validity_fake = out
        d_fake_loss = self.validity_loss_f(validity_fake, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()

        return recon_loss.data.mean(), adversarial_loss, validity_g_fake.data.mean(), \
               d_real_loss, validity_real.data.mean(), d_fake_loss, validity_fake.data.mean()

    def sample_latent_code(self, batch_size):
        return torch.from_numpy(self.latent_sampler.sample((batch_size, self.code_size))).type(FloatTensor)

    def _set_to_train(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

    def generate(self, mode, *args):
        self._set_to_eval()
        if mode == 'num':
            generator_input = self.sample_latent_code(args[0])
            gen_data = self.decoder(*generator_input)
        elif mode == 'fixed':
            gen_data = self.decoder(*args)
        else:
            raise ValueError('Unknown generator mode')
        self._set_to_train()
        return gen_data

    def reconstruct(self, images):
        self._set_to_eval()
        reconstruct_images = self.decoder(self.encoder(images))
        self._set_to_train()
        return reconstruct_images

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        state = {
            'net_d': self.discriminator.state_dict(),
            'net_g': self.encoder.state_dict(),
            'net_dec': self.decoder.state_dict(),
            'optimizer_d': self.optimizer_D.state_dict(),
            'optimizer_g': self.optimizer_G.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.discriminator.load_state_dict(checkpoint['net_d'])
        self.encoder.load_state_dict(checkpoint['net_g'])
        self.decoder.load_state_dict(checkpoint['net_dec'])
        if all:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_g'])
