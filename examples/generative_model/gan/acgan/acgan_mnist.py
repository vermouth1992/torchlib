"""
Train ACGAN on mnist
"""

import argparse
import pprint
import sys

import torch
import torch.nn as nn

from torchlib.utils.layers import conv2d_bn_lrelu_dropout_block, conv2d_trans_bn_lrelu_block
from torchlib.utils.weight import apply_weight_norm, weights_init_normal
from torchlib.dataset.image.mnist import get_mnist_data_loader
from torchlib.generative_model.gan.acgan.acgan import ACGAN
from torchlib.generative_model.gan.acgan.trainer import Trainer
from torchlib.generative_model.gan.acgan.utils import SampleImage, compute_accuracy


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
        self.label_embedding = nn.Embedding(10, 10)
        self.input_linear = nn.Linear(code_size + 10, 4 * 4 * self.inplane * 32)
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
        gen_input = torch.cat((self.label_embedding(args[1]), gen_input), -1)
        gen_input = self.input_linear(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.inplane * 32, 4, 4)
        img = self.model(gen_input)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACGAN for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    code_size = 100
    validity_loss_f = nn.MSELoss()
    class_loss_f = nn.CrossEntropyLoss()
    nb_classes = [10]
    lr_D = 2e-4
    lr_G = 1e-4
    checkpoint_path = './checkpoint/acgan_mnist.ckpt'
    weight_norm = None

    # models
    discriminator = Discriminator(weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr_D, betas=(0.5, 0.999))

    generator = Generator(code_size=code_size, weight_init=weights_init_normal, weight_norm=weight_norm)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr_G, betas=(0.5, 0.999))

    gan_model = ACGAN(generator, discriminator, optimizer_G, optimizer_D, None, None,
                      code_size, validity_loss_f, class_loss_f,
                      nb_classes)

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

        trainer.train(num_epoch, data_loader, gan_model, 1, checkpoint_path, 5, [sampler])

    else:
        # evaluate accuracy on test set
        gan_model.load_checkpoint(checkpoint_path, all=False)
        data_loader = get_mnist_data_loader(train=False)
        acc = compute_accuracy(data_loader, gan_model)
        print('The test accuracy of discriminator is {:.4f}'.format(acc[0]))
        # generate some samples
        import numpy as np
        from torch.utils.data import TensorDataset
        from torch.utils.data.dataloader import DataLoader
        labels = torch.from_numpy(np.random.randint(0, 10, (10000,)))
        samples = gan_model.generate('labels', labels)
        sample_data_loader = DataLoader(TensorDataset(samples, labels), batch_size=128)
        acc = compute_accuracy(sample_data_loader, gan_model)
        print('The synthetic accuracy of discriminator is {:.4f}'.format(acc[0]))

