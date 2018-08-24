"""
Auxiliary classifier GAN.
The generator takes image and a list of labels as input.
The discriminator takes image as input and produce 1. real/fake 2. a set of labels.
The main difference from CGAN is that during training of discriminator of real samples,
the label loss also needs to be considered.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from ....common import enable_cuda, FloatTensor, LongTensor


class ACGAN(object):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, optimizer_G: optim.Optimizer,
                 optimizer_D: optim.Optimizer, optimizer_G_scheduler: optim.lr_scheduler,
                 optimizer_D_scheduler: optim.lr_scheduler, code_size: int, validity_loss_f, class_loss_f,
                 nb_classes: list):
        assert type(nb_classes) == list, 'nb_classes must be a list for compatibility.'
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_G_scheduler = optimizer_G_scheduler
        self.optimizer_D_scheduler = optimizer_D_scheduler
        self.code_size = code_size
        self.validity_loss_f = validity_loss_f
        self.class_loss_f = class_loss_f
        self.nb_classes = nb_classes

        if enable_cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.class_loss_f.cuda()
            self.validity_loss_f.cuda()

    def _train_dis_with_real(self, real_data, real_labels, valid):
        self.optimizer_D.zero_grad()
        out = self.discriminator(real_data)
        validity_real = out[0]
        output_labels = out[1:]

        d_real_loss = self.validity_loss_f(validity_real, valid)
        for i in range(len(output_labels)):
            d_real_loss += self.class_loss_f(output_labels[i], real_labels[i])
        d_real_loss.backward()
        self.optimizer_D.step()
        return d_real_loss, validity_real.data.mean()

    def _train_dis_with_fake(self, z, gen_labels, fake):
        self.optimizer_D.zero_grad()
        generator_input = [z] + gen_labels
        gen_data = self.generator(*generator_input)
        out = self.discriminator(gen_data.detach())
        validity_fake = out[0]
        output_labels = out[1:]
        d_fake_loss = self.validity_loss_f(validity_fake, fake)
        for i in range(len(output_labels)):
            d_fake_loss += self.class_loss_f(output_labels[i], gen_labels[i])
        d_fake_loss.backward()
        self.optimizer_D.step()
        return d_fake_loss, validity_fake.data.mean()

    def _train_gen(self, z, gen_labels, valid):
        self.optimizer_G.zero_grad()
        generator_input = [z] + gen_labels
        gen_data = self.generator(*generator_input)
        out = self.discriminator(gen_data)
        validity_fake = out[0]
        output_labels = out[1:]
        g_loss = self.validity_loss_f(validity_fake, valid)
        for i in range(len(output_labels)):
            g_loss += self.class_loss_f(output_labels[i], gen_labels[i])
        g_loss.backward()
        self.optimizer_G.step()
        return g_loss, validity_fake.data.mean()

    def _set_to_train(self):
        self.generator.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def reset_grad(self):
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

    def update_parameters(self):
        self.optimizer_D.step()
        self.optimizer_G.step()

    def generate(self, mode, *args):
        self._set_to_eval()
        real_labels = []
        if mode == 'num':
            num_samples = args[0]
            z = Variable(FloatTensor(np.random.normal(0, 1, (num_samples, self.code_size))))
            for n in self.nb_classes:
                real_labels.append(Variable(LongTensor(np.random.randint(0, n, num_samples))))
            generator_input = tuple([z] + real_labels)
            gen_data = self.generator(*generator_input)
        elif mode == 'labels':
            assert len(args) == len(self.nb_classes), 'The number of arguments must equal to the number of labels'
            num_samples = args[0].shape[0]
            z = Variable(FloatTensor(np.random.normal(0, 1, (num_samples, self.code_size))))
            for arg in args:
                if type(arg) == np.ndarray:
                    arg = torch.from_numpy(arg)
                real_labels.append(Variable(arg.type(LongTensor)))
            generator_input = tuple([z] + real_labels)
            gen_data = self.generator(*generator_input)
        elif mode == 'fixed':
            z = Variable(FloatTensor(args[0]))
            for arg in args[1:]:
                if type(arg) == np.ndarray:
                    arg = torch.from_numpy(arg)
                real_labels.append(Variable(arg.type(LongTensor)))
            generator_input = tuple([z] + real_labels)
            gen_data = self.generator(*generator_input)
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
