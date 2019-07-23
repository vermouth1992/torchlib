import argparse
import pprint
from itertools import chain

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import transforms

from torchlib.dataset.image.cifar10 import get_cifar10_data_loader, get_cifar10_subset_data_loader
from torchlib.generative_model.autoencoder.utils import SampleImage, Reconstruction, VisualizeLatent
from torchlib.generative_model.autoencoder.vae import VAE, Trainer
from torchlib.utils.layers import conv2d_bn_lrelu_block, conv2d_trans_bn_lrelu_block


class Encoder(nn.Module):
    def __init__(self, code_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            *conv2d_bn_lrelu_block(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            *conv2d_bn_lrelu_block(32, 64, 4, 2, 1),
            *conv2d_bn_lrelu_block(64, 128, 4, 2, 1),
        )

        self.mu = nn.Linear(16 * 128, code_size)
        self.logvar = nn.Linear(16 * 128, code_size)

    def forward(self, img):
        x = self.model(img)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, code_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(code_size, 16 * 128)

        self.model = nn.Sequential(
            *conv2d_trans_bn_lrelu_block(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            *conv2d_trans_bn_lrelu_block(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            *conv2d_trans_bn_lrelu_block(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        img_flat = self.linear.forward(z)
        img = img_flat.view(img_flat.shape[0], 128, 4, 4)
        img = self.model(img)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for cifar10')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--resume', choices=['model', 'checkpoint'])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--code_size', type=int, default=10)
    args = vars(parser.parse_args())
    pprint.pprint(args)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # parameters
    train = args['train']
    code_size = args['code_size']
    recon_loss_f = nn.BCELoss(reduction='sum')
    checkpoint_path = './checkpoint/vae_cifar10.ckpt'
    learning_rate = 1e-3

    generator = Encoder(code_size)
    decoder = Decoder(code_size)
    optimizer = torch.optim.Adam(chain(generator.parameters(), decoder.parameters()), learning_rate)

    model = VAE(generator, decoder, code_size, optimizer)

    sampler = SampleImage(10, 10)
    data_loader = get_cifar10_subset_data_loader(train=True, transform=transform, fraction=100)
    reconstruct = Reconstruction(next(iter(data_loader))[0])

    summary_writer = SummaryWriter('runs/vae_cifar10')

    if train:
        visualize_data_loader = get_cifar10_subset_data_loader(train=True, transform=transform, fraction=1000)
        visualize_callback = VisualizeLatent(visualize_data_loader, method='pca')
        resume = args['resume']
        num_epoch = int(args['epoch'])

        if resume == 'model':
            model.load_checkpoint(checkpoint_path, all=False)
        elif resume == 'checkpoint':
            model.load_checkpoint(checkpoint_path, all=True)

        train_data_loader = get_cifar10_data_loader(train=True, transform=transform, batch_size=args['batch_size'])

        trainer = Trainer(recon_loss_f)
        trainer.train(num_epoch, train_data_loader, model, checkpoint_path, epoch_per_save=10,
                      callbacks=[sampler, reconstruct, visualize_callback], summary_writer=summary_writer)
