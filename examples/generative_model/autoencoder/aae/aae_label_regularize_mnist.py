"""
AAE on mnist
"""

import argparse
import pprint
import sys
from itertools import chain

import torch
import torch.nn as nn

from torchlib.generative_model.autoencoder.aae.aae_label_regularize.aae_label_regularize import AAELabelRegularize, \
    TrainerLabelRegularize
from torchlib.common import FloatTensor
from torchlib.dataset.image.mnist import get_mnist_data_loader, get_mnist_subset_data_loader
from torchlib.utils.random.sampler import ConditionGaussianSampler
from torchlib.generative_model.autoencoder.aae.utils import SampleImage, Reconstruction, VisualizeLatent


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = torch.randn_like(mu).type(FloatTensor)
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self, code_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32 * 32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, code_size)
        self.logvar = nn.Linear(512, code_size)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self, code_size):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(code_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 32 * 32 * 1),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], 1, 32, 32)
        return img


class Discriminator(nn.Module):
    def __init__(self, code_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(code_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AAE with label regularization for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    # parameters
    train = args['train']
    code_size = 5
    validity_loss_f = nn.BCEWithLogitsLoss()
    recon_loss_f = nn.L1Loss()
    lr_D = 2e-4
    lr_G = 1e-3
    checkpoint_path = './checkpoint/aae_label_regularize_mnist.ckpt'
    weight_norm = None
    alpha = 0.001

    # models
    discriminator = Discriminator(code_size)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr_D, betas=(0.5, 0.999))

    generator = Encoder(code_size)
    decoder = Decoder(code_size)
    optimizer_G = torch.optim.Adam(chain(generator.parameters(), decoder.parameters()), lr_G, betas=(0.5, 0.999))

    mu = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    sigma = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    gaussian_sampler = ConditionGaussianSampler(mu=mu, sigma=sigma)

    gan_model = AAELabelRegularize(generator, decoder, discriminator, optimizer_G, optimizer_D, code_size,
                                   validity_loss_f,
                                   recon_loss_f, gaussian_sampler, alpha=alpha)

    recon_data_loader = get_mnist_subset_data_loader(train=True, fraction=100)
    recon_callback = Reconstruction(recon_data_loader)

    sampler = SampleImage(10, 10)

    if train:
        visualize_data_loader = get_mnist_subset_data_loader(train=True, fraction=100)
        visualize_callback = VisualizeLatent(visualize_data_loader, method='pca')
        resume = args['resume']
        num_epoch = int(args['epoch'])

        if resume == 'model':
            gan_model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            gan_model.load_checkpoint(checkpoint_path, all=True)

        trainer = TrainerLabelRegularize(trick_dict={
            # 'label_smooth': {
            #     'valid_range': 0.95,
            #     'fake_range': 0
            # }
        })
        data_loader = get_mnist_data_loader(train=True)

        trainer.train(num_epoch, data_loader, gan_model, 1, checkpoint_path, 5,
                      [sampler, recon_callback, visualize_callback])

    else:
        # use notebook to visualize the latent distribution.
        visualize_data_loader = get_mnist_subset_data_loader(train=True, fraction=5000)
        visualize_callback = VisualizeLatent(visualize_data_loader, method='pca')
        gan_model.load_checkpoint(checkpoint_path, all=False)
        sampler(None, gan_model)
        recon_callback(None, gan_model)
        visualize_callback(None, gan_model)
