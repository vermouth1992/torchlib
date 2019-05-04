"""
Vanilla VAE model
"""

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchlib.common import FloatTensor
from torchlib.common import enable_cuda
from tqdm import tqdm


class VAE(object):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, code_size: int, optimizer: torch.optim.Optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.code_size = code_size
        self.optimizer = optimizer
        if enable_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def _set_to_train(self):
        self.encoder.train()
        self.decoder.train()

    def _set_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def sample_latent_code(self, batch_size):
        z = torch.randn(batch_size, self.code_size).type(FloatTensor)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        mu, logvar = self.encoder.forward(x)
        return mu, logvar

    def encode_reparm(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder.forward(z)

    def reconstruct(self, data):
        self._set_to_eval()
        with torch.no_grad():
            mu, logvar = self.encode(data)
            z = self.reparameterize(mu, logvar)
            return self.decode(z)

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        state = {
            'net_g': self.encoder.state_dict(),
            'net_dec': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['net_g'])
        self.decoder.load_state_dict(checkpoint['net_dec'])
        if all:
            self.optimizer.load_state_dict(checkpoint['optimizer'])


"""
Trainer for VAE
"""


class Trainer(object):
    def __init__(self, recon_loss_f):
        self.recon_loss_f = recon_loss_f

    def train(self, num_epoch, train_data_loader, model: VAE, checkpoint_path, epoch_per_save, callbacks,
              summary_writer: SummaryWriter):
        n_iter = 0
        for epoch in range(num_epoch):
            model._set_to_train()
            reconstruction_loss_train = 0.
            kl_divergence_train = 0.
            print('Epoch {}/{}'.format(epoch + 1, num_epoch))
            for input, label in tqdm(train_data_loader):
                model.optimizer.zero_grad()
                input = input.type(FloatTensor)
                mu, logvar = model.encode(input)
                z = model.reparameterize(mu, logvar)
                out = model.decode(z)

                reconstruction_loss = self.recon_loss_f(out, input)
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + kl_divergence
                loss.backward()
                model.optimizer.step()

                reconstruction_loss_train += reconstruction_loss.item()
                kl_divergence_train += kl_divergence.item()

                summary_writer.add_scalar('data/reconstruction_loss', reconstruction_loss.item(), n_iter)
                summary_writer.add_scalar('data/kl_divergence', kl_divergence.item(), n_iter)

                n_iter += 1

            reconstruction_loss_train /= len(train_data_loader.dataset)
            kl_divergence_train /= len(train_data_loader.dataset)

            print('Reconstruction loss: {:.4f} - KL divergence: {:.4f}'.format(reconstruction_loss_train,
                                                                               kl_divergence_train))

            if (epoch + 1) % epoch_per_save == 0:
                model.save_checkpoint(checkpoint_path)

            for callback in callbacks:
                callback(epoch, model, summary_writer)

        model.save_checkpoint(checkpoint_path)
