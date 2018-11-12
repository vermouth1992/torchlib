"""
Train WGAN on mnist dataset
"""

import argparse
import pprint
import sys

import torch
import torch.nn as nn

from torchlib.generative_model.gan.wgan.trainer import Trainer
from torchlib.generative_model.gan.wgan.wgan import WassersteinGAN
from torchlib.generative_model.gan.wgan.utils import SampleImage

from torchlib.utils.torch_layer_utils import conv2d_bn_lrelu_dropout_block, conv2d_trans_bn_lrelu_block
from torchlib.utils.weight_utils import weights_init_normal, apply_weight_norm
from torchlib.dataset.image.mnist import get_mnist_data_loader


class Discriminator(nn.Module):
    def __init__(self, weight_init=None, weight_norm=None):
        super(Discriminator, self).__init__()
        self.inplane = 64
        self.image_model = nn.Sequential(
            *conv2d_bn_lrelu_dropout_block(1, self.inplane, 3, 1, 1, normalize=False, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane, self.inplane * 2, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 2, self.inplane * 4, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 4, self.inplane * 8, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm)
        )

        self.fc_validity = nn.Linear(4 * 4 * self.inplane * 8, 1)

        if weight_init:
            self.apply(weight_init)

    def forward(self, img):
        img = self.image_model(img)
        d_in = img.view(img.size(0), -1)
        validity = self.fc_validity(d_in)
        return validity


class Generator(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None):
        super(Generator, self).__init__()
        self.inplane = 64
        self.input_linear = nn.Linear(code_size, 4 * 4 * self.inplane * 8)
        self.model = nn.Sequential(
            *conv2d_trans_bn_lrelu_block(self.inplane * 8, self.inplane * 4, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 4, self.inplane * 2, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 2, self.inplane, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            apply_weight_norm(nn.ConvTranspose2d(self.inplane, 1, 3, 1, 1, bias=False), weight_norm),
            nn.Tanh()
        )
        if weight_init:
            self.apply(weight_init)

    def forward(self, *args):
        gen_input = args[0]
        gen_input = self.input_linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.inplane * 8, 4, 4)
        img = self.model(gen_input)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WGAN for mnist')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    code_size = 64
    lr_D = 1e-4
    lr_G = 1e-4
    checkpoint_path = './checkpoint/wgan_mnist.ckpt'
    weight_norm = None
    validity_loss_f = nn.BCELoss()

    # models
    discriminator = Discriminator(weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr_D)

    generator = Generator(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr_G)

    gan_model = WassersteinGAN(generator, discriminator, optimizer_G, optimizer_D, code_size)

    sampler = SampleImage(10, code_size)

    if train:
        resume = args['resume']
        num_epoch = int(args['epoch'])

        if resume == 'model':
            gan_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            gan_model.load_checkpoint(checkpoint_path, all=True)

        trainer = Trainer(num_iter_D=5, clip=0.01)
        data_loader = get_mnist_data_loader(train=True)

        trainer.train(num_epoch, data_loader, gan_model, checkpoint_path, 5, [sampler])

    else:
        gan_model.load_checkpoint(checkpoint_path, all=False)
        sampler(None, gan_model)
