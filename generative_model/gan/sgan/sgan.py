"""
Semi-supervised GAN for image classification.
The output of D is valid/fake and n classes.
The discriminator is trying to maximize class label for real data and fake label for fake data.
The generator is trying to maximize any of the class label for fake data.
We want to use less than 10% of the data labels to train a classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim

from lib.common import enable_cuda, FloatTensor


class SemiSupervisedGAN(object):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, optimizer_G: optim.Optimizer,
                 optimizer_D: optim.Optimizer, optimizer_class: optim.Optimizer,
                 optimizer_G_scheduler: optim.lr_scheduler,
                 optimizer_D_scheduler: optim.lr_scheduler, optimizer_class_scheduler: optim.lr_scheduler,
                 code_size: int, validity_loss_f, class_loss_f, nb_classes: int):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_class = optimizer_class
        self.optimizer_G_scheduler = optimizer_G_scheduler
        self.optimizer_D_scheduler = optimizer_D_scheduler
        self.optimizer_class_scheduler = optimizer_class_scheduler
        self.code_size = code_size
        self.validity_loss_f = validity_loss_f
        self.class_loss_f = class_loss_f
        self.nb_classes = nb_classes

        if enable_cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.validity_loss_f.cuda()

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

    def _train_label(self, real_data, real_label):
        self.optimizer_class.zero_grad()
        out = self.discriminator(real_data)
        score = out[1]
        _, predicted = torch.max(score.data, 1)
        correct = (predicted == real_label).sum().item()
        d_class_loss = self.class_loss_f(score, real_label)
        d_class_loss.backward()
        self.optimizer_class.step()
        return d_class_loss.data.mean(), correct

    def sample_latent_code(self, batch_size):
        """ Always return a list for compatibility """
        z = torch.randn(batch_size, self.code_size).type(FloatTensor)
        return [z]

    def _set_to_train(self):
        self.generator.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.generator.eval()
        self.discriminator.eval()

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

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        state = {
            'net_d': self.discriminator.state_dict(),
            'net_g': self.generator.state_dict(),
            'optimizer_d': self.optimizer_D.state_dict(),
            'optimizer_g': self.optimizer_G.state_dict(),
        }
        if self.optimizer_D_scheduler:
            state['optimizer_d_scheduler'] = self.optimizer_D_scheduler.state_dict()
        if self.optimizer_G_scheduler:
            state['optimizer_g_scheduler'] = self.optimizer_G_scheduler.state_dict()
        if self.optimizer_class_scheduler:
            state['optimizer_class_scheduler'] = self.optimizer_class_scheduler.state_dict()
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.discriminator.load_state_dict(checkpoint['net_d'])
        self.generator.load_state_dict(checkpoint['net_g'])
        if all:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_g'])
            if self.optimizer_D_scheduler:
                self.optimizer_D_scheduler.load_state_dict(checkpoint['optimizer_d_scheduler'])
            if self.optimizer_G_scheduler:
                self.optimizer_G_scheduler.load_state_dict(checkpoint['optimizer_g_scheduler'])
            if self.optimizer_class_scheduler:
                self.optimizer_class_scheduler.load_state_dict(checkpoint['optimizer_class_scheduler'])
