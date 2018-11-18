"""
Train VAE on mnist dataset
"""

import argparse
import pprint
import sys
from itertools import chain

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchlib.dataset.image.mnist import get_mnist_data_loader, get_mnist_subset_data_loader
from torchlib.generative_model.autoencoder.utils import SampleImage, Reconstruction, VisualizeLatent
from torchlib.generative_model.autoencoder.vae import VAE, Trainer
from torchvision import transforms


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
        return mu, logvar


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
            nn.Sigmoid()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], 1, 32, 32)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AAE for MNIST')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', required='--train' in sys.argv)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])

    # parameters
    train = args['train']
    code_size = 10
    recon_loss_f = nn.BCELoss(reduction='sum')
    checkpoint_path = './checkpoint/vae_mnist.ckpt'
    learning_rate = 1e-3

    generator = Encoder(code_size)
    decoder = Decoder(code_size)
    optimizer = torch.optim.Adam(chain(generator.parameters(), decoder.parameters()), learning_rate)

    model = VAE(generator, decoder, code_size, optimizer)

    sampler = SampleImage(10, 10)
    data_loader = get_mnist_subset_data_loader(train=True, transform=transform, fraction=100)
    reconstruct = Reconstruction(next(iter(data_loader))[0])

    summary_writer = SummaryWriter('runs/vae_mnist')

    if train:
        visualize_data_loader = get_mnist_subset_data_loader(train=True, transform=transform, fraction=1000)
        visualize_callback = VisualizeLatent(visualize_data_loader, method='pca')
        resume = args['resume']
        num_epoch = int(args['epoch'])

        if resume == 'model':
            model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            model.load_checkpoint(checkpoint_path, all=True)

        train_data_loader = get_mnist_data_loader(train=True, transform=transform)

        trainer = Trainer(recon_loss_f)
        trainer.train(num_epoch, train_data_loader, model, checkpoint_path, epoch_per_save=10,
                      callbacks=[sampler, reconstruct, visualize_callback], summary_writer=summary_writer)
