"""
info GAN: https://arxiv.org/abs/1606.03657.
The latent variable of infoGAN is set randomly instead of the supervised way like ACGAN.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchlib.common import FloatTensor, LongTensor, enable_cuda, map_location
from torchlib.utils.random.torch_random_utils import uniform_tensor


class InfoGAN(object):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, optimizer_G: optim.Optimizer,
                 optimizer_D: optim.Optimizer, optimizer_info: optim.Optimizer,
                 optimizer_G_scheduler: optim.lr_scheduler, optimizer_D_scheduler: optim.lr_scheduler,
                 optimizer_info_scheduler: optim.lr_scheduler, code_size: int, latent: list, latent_weights: list,
                 validity_loss_f, latent_loss_f=None):
        """

        Args:
            generator: input: z and a list of latent code. output: image
            discriminator: input: image. output: real/fake and a list of latent code
            optimizer_G: optimizer for generator
            optimizer_D: optimizer for discriminator
            optimizer_G_scheduler: scheduler for optimizer G
            optimizer_D_scheduler: scheduler for optimizer D
            code_size: size for z
            latent: a list of number or tuple. Number indicates discrete latent and
                    (a, b) means range for continuous latent
            validity_loss_f: loss function for real/fake.
            latent_loss_f: a list loss function for each of the latent code. If none, we use
                           MSE for continuous latent code and cross entropy for discrete latent code.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_info = optimizer_info
        self.optimizer_G_scheduler = optimizer_G_scheduler
        self.optimizer_D_scheduler = optimizer_D_scheduler
        self.optimizer_info_scheduler = optimizer_info_scheduler
        self.code_size = code_size
        self.latent = latent
        self.latent_weights = latent_weights
        self.validity_loss_f = validity_loss_f
        if latent_loss_f is None:
            self.latent_loss_f = []
            for latent in self.latent:
                if isinstance(latent, int):
                    self.latent_loss_f.append(nn.CrossEntropyLoss())
                elif isinstance(latent, tuple):
                    self.latent_loss_f.append(nn.MSELoss())
                else:
                    raise ValueError('Unknown latent type')
        else:
            self.latent_loss_f = latent_loss_f

        if enable_cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.validity_loss_f.cuda()
            for latent_loss in self.latent_loss_f:
                latent_loss.cuda()

    def _train_dis_with_real(self, real_data, valid):
        self.optimizer_D.zero_grad()
        out = self.discriminator(real_data)
        validity_real = out[0]
        d_real_loss = self.validity_loss_f(validity_real, valid)
        d_real_loss.backward()
        self.optimizer_D.step()
        return d_real_loss, validity_real.data.mean()

    def _train_dis_with_fake(self, fake):
        self.optimizer_D.zero_grad()
        generator_input = self.sample_latent_code(fake.shape[0])
        gen_data = self.generator(*generator_input)
        out = self.discriminator(gen_data.detach())
        validity_fake = out[0]
        d_fake_loss = self.validity_loss_f(validity_fake, fake)
        d_fake_loss.backward()
        self.optimizer_D.step()
        return d_fake_loss, validity_fake.data.mean()

    def _train_gen(self, valid):
        self.optimizer_G.zero_grad()
        generator_input = self.sample_latent_code(valid.shape[0])
        gen_data = self.generator(*generator_input)
        out = self.discriminator(gen_data)
        validity_fake = out[0]
        g_loss = self.validity_loss_f(validity_fake, valid)
        g_loss.backward()
        self.optimizer_G.step()
        return g_loss, validity_fake.data.mean()

    def _train_info(self, batch_size):
        self.optimizer_info.zero_grad()
        generator_input = self.sample_latent_code(batch_size)
        latent_input = generator_input[1:]
        gen_data = self.generator(*generator_input)
        out = self.discriminator(gen_data)
        latent_output = out[1:]
        info_loss = None
        for i in range(len(latent_output)):
            if info_loss is None:
                info_loss = self.latent_loss_f[i](latent_output[i], latent_input[i]) * self.latent_weights[i]
            else:
                info_loss += self.latent_loss_f[i](latent_output[i], latent_input[i]) * self.latent_weights[i]
        info_loss.backward()
        self.optimizer_info.step()
        return info_loss

    def generate(self, mode, *args):
        self._set_to_eval()
        if mode == 'num':
            generator_input = tuple(self.sample_latent_code(args[0]))
            gen_data = self.generator(*generator_input)
        elif mode == 'fixed':
            gen_data = self.generator(*args)
        else:
            raise ValueError('Unknown generator mode')
        self._set_to_train()
        return gen_data

    def sample_latent_code(self, batch_size):
        z = []
        z.append(torch.randn(batch_size, self.code_size).type(FloatTensor))
        for latent in self.latent:
            if isinstance(latent, int):
                data = torch.randint(0, latent, (batch_size,)).type(LongTensor)
            elif isinstance(latent, tuple):
                data = uniform_tensor(batch_size, 1, r1=latent[0], r2=latent[1]).type(FloatTensor)
            else:
                raise ValueError('Unknown latent type')
            z.append(data)
        return z

    def _set_to_train(self):
        self.generator.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        state = {
            'net_d': self.discriminator.state_dict(),
            'net_g': self.generator.state_dict(),
            'optimizer_d': self.optimizer_D.state_dict(),
            'optimizer_g': self.optimizer_G.state_dict(),
            'optimizer_info': self.optimizer_info.state_dict()
        }
        if self.optimizer_D_scheduler:
            state['optimizer_d_scheduler'] = self.optimizer_D_scheduler.state_dict()
        if self.optimizer_G_scheduler:
            state['optimizer_g_scheduler'] = self.optimizer_G_scheduler.state_dict()
        if self.optimizer_info_scheduler:
            state['optimizer_info_scheduler'] = self.optimizer_info_scheduler.state_dict()
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.discriminator.load_state_dict(checkpoint['net_d'])
        self.generator.load_state_dict(checkpoint['net_g'])
        if all:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_g'])
            self.optimizer_info.load_state_dict(checkpoint['optimizer_info'])
            if self.optimizer_D_scheduler:
                self.optimizer_D_scheduler.load_state_dict(checkpoint['optimizer_d_scheduler'])
            if self.optimizer_G_scheduler:
                self.optimizer_G_scheduler.load_state_dict(checkpoint['optimizer_g_scheduler'])
            if self.optimizer_info_scheduler:
                self.optimizer_info_scheduler.load_state_dict(checkpoint['optimizer_info_scheduler'])
