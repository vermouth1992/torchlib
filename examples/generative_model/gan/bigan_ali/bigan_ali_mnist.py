"""
We use the architecture in the paper. https://arxiv.org/abs/1606.00704
"""

import argparse
import pprint
import sys
from itertools import chain

import torch
import torch.nn as nn

from torchlib.common import FloatTensor
from torchlib.generative_model.gan.bigan_ali.bigan_ali import ALI
from torchlib.dataset.image.mnist import get_mnist_data_loader, get_mnist_subset_data_loader
from torchlib.utils.layers import conv2d_bn_lrelu_dropout_block, conv2d_trans_bn_lrelu_block, \
    linear_bn_lrelu_dropout_block, conv2d_bn_lrelu_block
from torchlib.utils.weight import apply_weight_norm, weights_init_normal
from torchlib.contrib.minibatch_discrimination import MinibatchDiscrimination
from torchlib.generative_model.gan.bigan_ali.trainer import Trainer
from torchlib.generative_model.gan.bigan_ali.utils import SampleImage, Reconstruction, VisualizeLatent


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = torch.randn_like(mu).type(FloatTensor)
    z = sampled_z * std + mu
    return z

class Discriminator(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None, minibatch_norm=False):
        super(Discriminator, self).__init__()
        self.inplane = 16
        self.image_model = nn.Sequential(
            *conv2d_bn_lrelu_dropout_block(1, self.inplane, 3, 1, 1, normalize=False, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane, self.inplane * 2, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 2, self.inplane * 4, 3, 1, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 4, self.inplane * 8, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 8, self.inplane * 16, 3, 1, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm),
            *conv2d_bn_lrelu_dropout_block(self.inplane * 16, self.inplane * 32, 4, 2, 1, normalize=True, bias=False,
                                           weight_norm=weight_norm)
        )
        self.noise_model = nn.Sequential(
            *linear_bn_lrelu_dropout_block(code_size, 256),
            *linear_bn_lrelu_dropout_block(256, 256)
        )
        self.minibatch_norm = minibatch_norm

        if minibatch_norm:
            self.minibatch_norm = MinibatchDiscrimination(4 * 4 * self.inplane * 32 + 256, 256, kernel_dims=30)
            self.fc_validity = nn.Linear(4 * 4 * self.inplane * 32 + 256 + 256, 1)
        else:
            self.fc_validity = nn.Linear(4 * 4 * self.inplane * 32 + 256, 1)

        self.activation = nn.Sigmoid()
        if weight_init:
            self.apply(weight_init)

    def forward(self, img, z):
        img = self.image_model(img)
        d_in_image = img.view(img.size(0), -1)
        d_in_z = self.noise_model(z)
        d_in = torch.cat((d_in_image, d_in_z), -1)
        if self.minibatch_norm:
            d_in = self.minibatch_norm(d_in)
        validity = self.activation(self.fc_validity(d_in))
        return validity


class Generator(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None):
        super(Generator, self).__init__()
        self.inplane = 8
        self.input_linear = nn.Linear(code_size, 4 * 4 * self.inplane * 32)
        self.model = nn.Sequential(
            *conv2d_trans_bn_lrelu_block(self.inplane * 32, self.inplane * 32, 3, 1, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 32, self.inplane * 16, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 16, self.inplane * 16, 3, 1, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 16, self.inplane * 8, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 8, self.inplane * 8, 3, 1, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 8, self.inplane * 4, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            apply_weight_norm(nn.ConvTranspose2d(self.inplane * 4, 1, 3, 1, 1, bias=False), weight_norm),
            nn.Tanh()
        )
        if weight_init:
            self.apply(weight_init)

    def forward(self, *args):
        gen_input = args[0]
        gen_input = self.input_linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.inplane * 32, 4, 4)
        img = self.model(gen_input)
        return img


class Encoder(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_lrelu_block(1, 32, 3, 1, 1, normalize=True, bias=False, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(32, 64, 4, 2, 1, normalize=True, bias=False, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(64, 128, 3, 1, 1, normalize=True, bias=False, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(128, 256, 4, 2, 1, normalize=True, bias=False, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(256, 512, 3, 1, 1, normalize=True, bias=False, weight_norm=weight_norm),
            *conv2d_bn_lrelu_block(512, 512, 4, 2, 1, normalize=True, bias=False, weight_norm=weight_norm)
        )

        self.linear_mu = nn.Linear(512 * 4 * 4, code_size)
        self.linear_log_var = nn.Linear(512 * 4 * 4, code_size)

        if weight_init:
            self.apply(weight_init)

    def forward(self, *input):
        img = self.model(input[0])
        img = img.view(img.size(0), -1)
        mu = self.linear_mu(img)
        log_var = self.linear_log_var(img)
        z = reparameterization(mu, log_var)
        return z


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiGAN/ALI for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    resume = args['resume']
    num_epoch = int(args['epoch'])
    code_size = 10
    validity_loss_f = nn.BCELoss()
    lr_D = 1e-4
    lr_G = 1e-4
    minibatch_norm = False
    checkpoint_path = './checkpoint/bigan_ali_mnist.ckpt'
    weight_norm = None

    # models
    discriminator = Discriminator(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr_D, betas=(0.5, 0.999))

    generator = Generator(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    encoder = Encoder(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_G = torch.optim.Adam(chain(generator.parameters(), encoder.parameters()), lr_G, betas=(0.5, 0.999))

    gan_model = ALI(generator, encoder, discriminator, optimizer_G, optimizer_D, code_size, validity_loss_f)

    mnist_subset = get_mnist_subset_data_loader(train=True, fraction=100)

    sampler = SampleImage(10, code_size)
    reconstruction = Reconstruction(mnist_subset)
    visualize_latent = VisualizeLatent(mnist_subset, method='pca')

    if train:

        if resume == 'model':
            gan_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            gan_model.load_checkpoint(checkpoint_path, all=True)

        trainer = Trainer(trick_dict={
            'label_smooth': {
                'valid_range': 0.95,
                'fake_range': 0
            }
        })
        data_loader = get_mnist_data_loader(train=True, batch_size=128)

        trainer.train(num_epoch, data_loader, gan_model, checkpoint_path, 5, [sampler, reconstruction, visualize_latent])

    else:
        # use notebook to perform downstream tasks and compare the result with acgan. But MNIST is trivial
        pass
