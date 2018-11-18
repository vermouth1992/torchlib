"""
Adversarial Activated VAE
"""

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchlib.common import FloatTensor
from torchlib.common import enable_cuda
from torchlib.utils.torch_layer_utils import freeze, unfreeze
from tqdm import tqdm


class AAVAE(object):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 code_size: int, optimizer_G: torch.optim.Optimizer,
                 optimizer_D: torch.optim.Optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.code_size = code_size
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        if enable_cuda:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()

    def __repr__(self):
        return 'aavae'

    def _set_to_train(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

    def _set_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()

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
            'net_d': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_G.state_dict(),
            'optimizer_d': self.optimizer_D.state_dict()
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, all=True):
        print('Loading checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['net_g'])
        self.decoder.load_state_dict(checkpoint['net_dec'])
        self.discriminator.load_state_dict(checkpoint['net_d'])
        if all:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_g'])


"""
Trainer for AAVAE
"""


class Trainer(object):
    def __init__(self, recon_loss_f, validity_loss_f):
        self.recon_loss_f = recon_loss_f
        self.validity_loss_f = validity_loss_f

    def train(self, num_epoch, train_data_loader, model: AAVAE, checkpoint_path, epoch_per_save, callbacks,
              summary_writer: SummaryWriter, adv_ratio):
        n_iter = 0
        for epoch in range(num_epoch):
            model._set_to_train()
            reconstruction_loss_train = 0.
            kl_divergence_train = 0.
            print('Epoch {}/{}'.format(epoch + 1, num_epoch))
            for input, label in tqdm(train_data_loader):
                batch_size = input.shape[0]
                # train generator
                freeze(model.discriminator)
                unfreeze(model.encoder)
                unfreeze(model.decoder)
                model.optimizer_G.zero_grad()
                input = input.type(FloatTensor)
                mu, logvar = model.encode(input)
                z = model.reparameterize(mu, logvar)
                out = model.decode(z)

                reconstruction_loss = self.recon_loss_f(out, input)
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                valid = Variable(FloatTensor(batch_size, 1).fill_(1.), requires_grad=False)

                validity = model.discriminator(out)

                adversarial_loss = self.validity_loss_f(validity, valid)

                loss = reconstruction_loss + kl_divergence + adversarial_loss * adv_ratio
                loss.backward()
                model.optimizer_G.step()

                reconstruction_loss_train += reconstruction_loss.item()
                kl_divergence_train += kl_divergence.item()

                summary_writer.add_scalar('data/reconstruction_loss', reconstruction_loss.item(), n_iter)
                summary_writer.add_scalar('data/kl_divergence', kl_divergence.item(), n_iter)

                # train discriminator
                unfreeze(model.discriminator)
                freeze(model.encoder)
                freeze(model.decoder)
                model.optimizer_D.zero_grad()
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.), requires_grad=False)

                validity_real = model.discriminator(input)
                d_real_loss = self.validity_loss_f(validity_real, valid)
                validity_fake = model.discriminator(out.detach())
                d_fake_loss = self.validity_loss_f(validity_fake, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                model.optimizer_D.step()

                summary_writer.add_scalars('data/adversarial_loss', {'g_loss': adversarial_loss.item(),
                                                                     'd_loss': d_loss.item()}, n_iter)

                summary_writer.add_scalars('data/adversarial_accuracy', {'real': validity_real.data.mean(),
                                                                         'fake': validity_fake.data.mean()},
                                           n_iter)

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
