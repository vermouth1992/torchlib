"""
Semi supervised GAN on MNIST
"""

import argparse
import pprint
import sys
from itertools import chain

import torch
import torch.nn as nn

from torchlib.dataset.image.mnist import get_mnist_data_loader, get_mnist_subset_data_loader
from torchlib.utils.layers import conv2d_bn_lrelu_dropout_block, conv2d_trans_bn_lrelu_block
from torchlib.utils.weight import apply_weight_norm, weights_init_normal
from torchlib.generative_model.gan.sgan.sgan import SemiSupervisedGAN
from torchlib.generative_model.gan.sgan.trainer import Trainer
from torchlib.generative_model.gan.sgan.utils import compute_accuracy, SampleImage


class Discriminator(nn.Module):
    def __init__(self, weight_init=None, weight_norm=None):
        super(Discriminator, self).__init__()
        self.inplane = 16
        self.model = nn.Sequential(
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
        self.fc_validity = nn.Linear(4 * 4 * self.inplane * 32, 1)
        self.fc_class = nn.Linear(4 * 4 * self.inplane * 32, 10)
        if weight_init:
            self.apply(weight_init)

    def forward(self, img):
        img = self.model(img)
        d_in = img.view(img.size(0), -1)
        validity = self.fc_validity(d_in)
        score = self.fc_class(d_in)
        return validity, score


class Generator(nn.Module):
    def __init__(self, code_size, weight_init=None, weight_norm=None):
        super(Generator, self).__init__()
        self.inplane = 8
        self.input_linear = nn.Linear(code_size, 4 * 4 * self.inplane * 32)
        self.model = nn.Sequential(
            *conv2d_trans_bn_lrelu_block(self.inplane * 32, self.inplane * 16, 4, 2, 1, normalize=True, bias=False,
                                         weight_norm=weight_norm),
            *conv2d_trans_bn_lrelu_block(self.inplane * 16, self.inplane * 8, 4, 2, 1, normalize=True, bias=False,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGAN for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    code_size = 100
    validity_loss_f = nn.BCEWithLogitsLoss()
    class_loss_f = nn.CrossEntropyLoss()
    nb_classes = 10
    lr_D = 2e-4
    lr_G = 1e-4
    checkpoint_path = './checkpoint/sgan_mnist.ckpt'
    weight_norm = None
    fraction = 100

    # models
    discriminator = Discriminator(weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_D = torch.optim.Adam(chain(discriminator.model.parameters(), discriminator.fc_validity.parameters()),
                                   lr_D, betas=(0.5, 0.999))

    optimizer_class = torch.optim.Adam(discriminator.fc_class.parameters(), lr_D, betas=(0.5, 0.999))

    generator = Generator(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr_G, betas=(0.5, 0.999))

    gan_model = SemiSupervisedGAN(generator, discriminator, optimizer_G, optimizer_D, optimizer_class,
                                  None, None, None, code_size, validity_loss_f, class_loss_f, nb_classes)

    if train:
        resume = args['resume']
        num_epoch = int(args['epoch'])
        sampler = SampleImage(10, 10, code_size)
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
        data_loader = get_mnist_data_loader(train=True)
        data_loader_label = get_mnist_subset_data_loader(train=True, fraction=fraction)

        trainer.train(num_epoch, data_loader, data_loader_label, gan_model, 1, checkpoint_path, 5, [sampler])

    else:
        # evaluate accuracy on test set
        gan_model.load_checkpoint(checkpoint_path, all=False)
        data_loader = get_mnist_data_loader(train=False)
        acc = compute_accuracy(data_loader, gan_model)
        print('The test accuracy of discriminator is {:.4f}'.format(acc))
