"""
Vanilla VAE model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from torchlib.common import enable_cuda, move_tensor_to_gpu
from torchlib.utils.math import log_to_log2


class VAE(object):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, prior: Distribution,
                 optimizer: torch.optim.Optimizer):
        """

        Args:
            encoder: The encoder must output a Pytorch distribution.
            decoder: Pytorch module takes in latent code and output a sample distribution.
            prior: p(z)
            optimizer: optimizer of the network
        """
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
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
        return self.prior.sample(torch.Size((batch_size,)))

    def encode(self, x):
        return self.encoder.forward(x)

    def encode_reparm(self, x):
        return self.encode(x).rsample()

    def decode(self, z):
        return self.decode_distribution(z).mean

    def decode_distribution(self, z):
        return self.decoder.forward(z)

    @torch.no_grad()
    def sample(self, batch_size, full_path=False):
        self._set_to_eval()
        z = self.sample_latent_code(batch_size=batch_size)
        decode_distribution = self.decode_distribution(z)
        if full_path:
            return decode_distribution.sample()
        else:
            return decode_distribution.mean

    @torch.no_grad()
    def reconstruct(self, data):
        self._set_to_eval()
        latent_distribution = self.encode(data)
        z = latent_distribution.sample()
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

    def train(self, num_epoch, train_data_loader, checkpoint_path=None, epoch_per_save=5, callbacks=(),
              summary_writer: SummaryWriter = None, verbose=True):
        n_iter = 0
        for epoch in range(num_epoch):
            self._set_to_train()
            negative_log_likelihood_train = 0.
            kl_divergence_train = 0.

            if verbose:
                t = tqdm(train_data_loader, desc='Epoch {}/{}'.format(epoch + 1, num_epoch))
            else:
                t = train_data_loader

            for data_batch in t:
                input = data_batch[0]
                self.optimizer.zero_grad()
                input = move_tensor_to_gpu(input)
                latent_distribution = self.encode(input)
                z = latent_distribution.rsample()
                out = self.decode_distribution(z)

                negative_log_likelihood = -out.log_prob(input).sum()
                kl_divergence = torch.distributions.kl_divergence(latent_distribution, self.prior).sum()

                loss = negative_log_likelihood + kl_divergence
                loss.backward()
                self.optimizer.step()

                negative_log_likelihood_train += negative_log_likelihood.item()
                kl_divergence_train += kl_divergence.item()

                if summary_writer:
                    summary_writer.add_scalar('data/nll', negative_log_likelihood.item(), n_iter)
                    summary_writer.add_scalar('data/kld', kl_divergence.item(), n_iter)

                n_iter += 1

            if verbose:
                num_dimensions = np.prod(list(train_data_loader.dataset[0][0].shape))
                negative_log_likelihood_train /= len(train_data_loader.dataset)
                negative_log_likelihood_train_bits_per_dim = log_to_log2(negative_log_likelihood_train / num_dimensions)
                kl_divergence_train /= len(train_data_loader.dataset)
                kl_divergence_train_bits_per_dim = log_to_log2(kl_divergence_train / num_dimensions)
                total_loss = negative_log_likelihood_train + kl_divergence_train
                total_loss_bits_per_dim = log_to_log2(total_loss / num_dimensions)

                total_loss_message = 'Totol loss {:.4f}/{:.4f} (bits/dim)'.format(total_loss, total_loss_bits_per_dim)
                nll_message = 'Negative log likelihood {:.4f}/{:.4f} (bits/dim)'.format(
                    negative_log_likelihood_train, negative_log_likelihood_train_bits_per_dim)
                kl_message = 'KL divergence {:.4f}/{:.4f} (bits/dim)'.format(kl_divergence_train,
                                                                             kl_divergence_train_bits_per_dim)

                print(' - '.join([total_loss_message, nll_message, kl_message]))

            if checkpoint_path is not None and (epoch + 1) % epoch_per_save == 0:
                self.save_checkpoint(checkpoint_path)

            if summary_writer:
                for callback in callbacks:
                    callback(epoch, self, summary_writer)
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
