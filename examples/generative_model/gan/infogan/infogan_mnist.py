"""
Train info GAN on MNIST. We use 1 discrete and 2 continuous latent code as described in the paper.
"""

import argparse
import pprint
import sys

import numpy as np
import torch
import torch.nn as nn
from visdom import Visdom
from itertools import chain

from torchlib.generative_model.gan.infogan.infogan import InfoGAN
from torchlib.common import LongTensor, FloatTensor
from torchlib.dataset.image.mnist import get_mnist_data_loader
from torchlib.utils.torch_layer_utils import conv2d_bn_lrelu_block, conv2d_trans_bn_relu_block
from torchlib.utils.weight_utils import apply_weight_norm, weights_init_normal
from torchlib.generative_model.gan.infogan.trainer import Trainer


class Discriminator(nn.Module):
    def __init__(self, latent_lst, weight_init=None, weight_norm=None):
        super(Discriminator, self).__init__()
        self.inplane = 16
        self.model = nn.Sequential(
            *conv2d_bn_lrelu_block(1, self.inplane, 3, 1, 1, normalize=False, bias=False,
                                   weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(self.inplane, self.inplane * 2, 4, 2, 1, normalize=False, bias=False,
                                   weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(self.inplane * 2, self.inplane * 4, 3, 1, 1, normalize=False, bias=False,
                                   weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(self.inplane * 4, self.inplane * 8, 4, 2, 1, normalize=False, bias=False,
                                   weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(self.inplane * 8, self.inplane * 16, 3, 1, 1, normalize=False, bias=False,
                                   weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(self.inplane * 16, self.inplane * 32, 4, 2, 1, normalize=False, bias=False,
                                   weight_norm=weight_norm)
        )
        self.fc_validity = nn.Linear(4 * 4 * self.inplane * 32, 1)
        self.latent_fc = nn.ModuleList()
        for latent in latent_lst:
            if isinstance(latent, int):
                self.latent_fc.append(nn.Linear(4 * 4 * self.inplane * 32, latent))
            elif isinstance(latent, tuple):
                self.latent_fc.append(nn.Linear(4 * 4 * self.inplane * 32, 1))
            else:
                raise ValueError('Unknown latent type')

        if weight_init:
            self.apply(weight_init)

    def forward(self, *input):
        img = self.model(input[0])
        d_in = img.view(img.size(0), -1)
        validity = self.fc_validity(d_in)
        score = [validity]
        for fc in self.latent_fc:
            score.append(fc(d_in))
        return score


class Generator(nn.Module):
    def __init__(self, latent_lst, code_size, weight_init=None, weight_norm=None):
        super(Generator, self).__init__()
        self.inplane = 8
        self.label_embedding = nn.ModuleList()
        input_size = code_size
        for latent in latent_lst:
            if isinstance(latent, int):
                self.label_embedding.append(nn.Embedding(latent, latent))
                input_size += latent
            else:
                self.label_embedding.append(None)
                input_size += 1
        self.input_linear = nn.Linear(input_size, 4 * 4 * self.inplane * 32)
        self.model = nn.Sequential(
            *conv2d_trans_bn_relu_block(self.inplane * 32, self.inplane * 16, 4, 2, 1, normalize=True, bias=False,
                                        weight_norm=weight_norm),
            *conv2d_trans_bn_relu_block(self.inplane * 16, self.inplane * 8, 4, 2, 1, normalize=True, bias=False,
                                        weight_norm=weight_norm),
            *conv2d_trans_bn_relu_block(self.inplane * 8, self.inplane * 4, 4, 2, 1, normalize=True, bias=False,
                                        weight_norm=weight_norm),
            apply_weight_norm(nn.ConvTranspose2d(self.inplane * 4, 1, 3, 1, 1, bias=False), weight_norm),
            nn.Tanh()
        )
        if weight_init:
            self.apply(weight_init)

    def forward(self, *args):
        z = args[0]
        gen_input = list(args[1:])
        for i, input in enumerate(gen_input):
            if self.label_embedding[i] is not None:
                gen_input[i] = self.label_embedding[i](input)
        gen_input = [z] + gen_input
        gen_input = torch.cat(gen_input, -1)
        gen_input = self.input_linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.inplane * 32, 4, 4)
        img = self.model(gen_input)
        return img


class SampleImage(object):
    """ Two continuous variables. One discrete variable. """

    def __init__(self, latent_bound=2.):
        self.viz = Visdom()
        self.fixed_z = torch.from_numpy(np.random.normal(0, 1, (100, code_size))).type(FloatTensor)
        self.labels = torch.from_numpy(np.array([num for num in range(10) for _ in range(10)])).type(LongTensor)
        self.fixed_latent = torch.from_numpy(np.expand_dims(np.array([0. for _ in range(100)]), axis=1)).type(
            FloatTensor)
        self.latent = torch.from_numpy(
            np.expand_dims(np.tile(np.linspace(-latent_bound, latent_bound, 10), 10), axis=1)).type(FloatTensor)
        self.win1 = None
        self.win2 = None
        self.n_row = 10

    def __call__(self, trainer, gan_model):
        if trainer:
            global_step = trainer.global_step
        else:
            global_step = 0
        output_images_1 = (gan_model.generate('fixed', self.fixed_z, self.labels, self.fixed_latent,
                                              self.latent) + 1.) / 2.

        output_images_2 = (gan_model.generate('fixed', self.fixed_z, self.labels, self.latent,
                                              self.fixed_latent) + 1.) / 2.

        if self.win1 is None:
            self.win1 = self.viz.images(output_images_1, nrow=self.n_row, opts=dict(
                caption='Step: {}'.format(global_step)
            ))
        else:
            self.viz.images(output_images_1, nrow=self.n_row, win=self.win1, opts=dict(
                caption='Step: {}'.format(global_step)
            ))

        if self.win2 is None:
            self.win2 = self.viz.images(output_images_2, nrow=self.n_row, opts=dict(
                caption='Step: {}'.format(global_step)
            ))
        else:
            self.viz.images(output_images_2, nrow=self.n_row, win=self.win2, opts=dict(
                caption='Step: {}'.format(global_step)
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infogan for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    code_size = 100
    latent = [10, (-1, 1), (-1, 1)]
    latent_weights = [1, 1, 1]
    validity_loss_f = nn.BCEWithLogitsLoss()
    lr_D = 2e-4
    lr_G = 1e-3
    checkpoint_path = './checkpoint/infogan_mnist.ckpt'
    weight_norm = None

    # models
    discriminator = Discriminator(latent, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_D = torch.optim.Adam(chain(discriminator.model.parameters(), discriminator.fc_validity.parameters()),
                                   lr_D, betas=(0.5, 0.999))

    generator = Generator(latent, code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr_G, betas=(0.5, 0.999))

    optimizer_info = torch.optim.Adam(chain(discriminator.latent_fc.parameters(), generator.parameters()), lr_D,
                                      betas=(0.5, 0.999))

    gan_model = InfoGAN(generator, discriminator, optimizer_G, optimizer_D, optimizer_info, None, None, None, code_size,
                        latent,
                        latent_weights, validity_loss_f)

    if train:
        resume = args['resume']
        num_epoch = int(args['epoch'])
        sampler = SampleImage()
        if resume == 'model':
            gan_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            gan_model.load_checkpoint(checkpoint_path, all=True)

        trainer = Trainer(trick_dict={
            'label_smooth': {
                'valid_range': 0.95,
                'fake_range': 0
            },
            # 'noisy_input': {
            #     'sigma': 0.1,
            #     'decay': 2e-6
            # },
            # 'flip_label': {
            #     'num_steps_per_flip': 30
            # }
        })
        data_loader = get_mnist_data_loader(train=True)

        trainer.train(num_epoch, data_loader, gan_model, 1, checkpoint_path, 5, [sampler])

    else:
        # evaluate accuracy on test set
        gan_model.load_checkpoint(checkpoint_path, all=False)
        data_loader = get_mnist_data_loader(train=False)
        sampler = SampleImage(latent_bound=2.)
        sampler(None, gan_model)
