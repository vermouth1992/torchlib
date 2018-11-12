"""
Train WGAN on cifar10
"""

import argparse
import pprint
import sys

import torch
import torch.nn as nn

from torchlib.generative_model.gan.wgan.trainer import Trainer
from torchlib.generative_model.gan.wgan.wgan import WassersteinGAN
from torchlib.generative_model.gan.wgan.utils import SampleImage

from torchlib.utils.torch_layer_utils import linear_bn_lrelu_dropout_block
from torchlib.utils.weight_utils import weights_init_normal
from torchlib.dataset.image.cifar10 import get_cifar10_data_loader


class Discriminator(nn.Module):
    def __init__(self, weight_init=None, weight_norm=None):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *linear_bn_lrelu_dropout_block(32 * 32 * 3, 512),
            *linear_bn_lrelu_dropout_block(512, 512),
            *linear_bn_lrelu_dropout_block(512, 512),
            nn.Linear(512, 1)
        )

        self.apply(weight_init)

    def forward(self, *input):
        img = input[0]
        img = img.view(img.size(0), -1)
        return self.model(img)


class Generator(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            *linear_bn_lrelu_dropout_block(code_size, 512),
            *linear_bn_lrelu_dropout_block(512, 512),
            *linear_bn_lrelu_dropout_block(512, 512),
            nn.Linear(512, 32 * 32 * 3),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, *input):
        img = self.model(input[0])
        img = img.view(img.size(0), 3, 32, 32)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WGAN for cifar10')
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
    checkpoint_path = './checkpoint/wgan_cifar10.ckpt'
    weight_norm = None

    # models
    discriminator = Discriminator(weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr_D)

    generator = Generator(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr_G)

    gan_model = WassersteinGAN(generator, discriminator, optimizer_G, optimizer_D, code_size)

    if train:
        resume = args['resume']
        num_epoch = int(args['epoch'])
        sampler = SampleImage(10, code_size)
        if resume == 'model':
            gan_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            gan_model.load_checkpoint(checkpoint_path, all=True)

        trainer = Trainer(num_iter_D=5, clip=0.01)
        data_loader = get_cifar10_data_loader(train=True)

        trainer.train(num_epoch, data_loader, gan_model, checkpoint_path, 5, [sampler])

    else:
        pass
