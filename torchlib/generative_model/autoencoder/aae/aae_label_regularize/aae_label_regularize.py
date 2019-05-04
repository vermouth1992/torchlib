"""
Label regularized AAE.
The only difference is that when matching latent distribution, it will be conditioned on the class label.
"""

import numpy as np
import torch
from tqdm import tqdm

from ..aae.aae import AAE
from ..aae.aae_trainer import Trainer
from lib.common import FloatTensor


class AAELabelRegularize(AAE):
    def sample_latent_code(self, arg):
        """

        Args:
            args: can be a label tensor or a number

        Returns:

        """
        if type(arg) == int:
            # random select
            labels = np.random.randint(0, 10, size=arg)
        else:
            labels = arg.cpu().numpy()
        batch_size = labels.shape[0]
        return torch.from_numpy(self.latent_sampler.sample((batch_size, self.code_size), labels)).type(FloatTensor)

    def _train_step(self, images, valid, fake):
        images, labels = images

        self.optimizer_G.zero_grad()
        # forward pass
        z = self.encoder(images)
        recon_images = self.decoder(z)
        recon_loss = self.recon_loss_f(recon_images, images)

        out = self.discriminator(z)
        validity_g_fake = out
        adversarial_loss = self.validity_loss_f(validity_g_fake, valid)

        g_loss = self.alpha * adversarial_loss + (1 - self.alpha) * recon_loss
        g_loss.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        real_data = self.sample_latent_code(labels)
        out = self.discriminator(real_data)
        validity_real = out
        d_real_loss = self.validity_loss_f(validity_real, valid)

        out = self.discriminator(z.detach())
        validity_fake = out
        d_fake_loss = self.validity_loss_f(validity_fake, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()

        return recon_loss.data.mean(), adversarial_loss, validity_g_fake.data.mean(), \
               d_real_loss, validity_real.data.mean(), d_fake_loss, validity_fake.data.mean()


class TrainerLabelRegularize(Trainer):
    def train(self, num_epoch, data_loader, gan_model: AAE, disc_iter, checkpoint_path, epoch_per_save,
              callbacks):
        assert disc_iter > 0, 'Discriminator update iteration must be greater than zero'
        for epoch in range(num_epoch):
            gan_model._set_to_train()
            # we sample a batch after each epoch
            dis_loss_lst = []
            gen_loss_lst = []
            recon_loss_lst = []
            D_x_lst = []
            D_G_z1_lst = []
            D_G_z2_lst = []
            # plot smoothing
            smooth_factor = 0.95
            plot_dis_s = 0
            plot_gen_s = 0
            plot_D_x = 0
            plot_D_G_z1 = 0
            plot_D_G_z2 = 0
            plot_ws = 0

            print('Epoch {}'.format(epoch + 1))
            for input_and_aux in tqdm(data_loader):
                # We assume the input_and_label is a tuple containing data and auxiliary information
                # train gan
                batch_size = input_and_aux[0].shape[0]
                valid = self._create_valid(batch_size)
                fake = self._create_fake(batch_size)

                # train ae
                recon_loss, g_loss, D_G_z2, d_real_loss, D_x, d_fake_loss, D_G_z1 = gan_model._train_step(input_and_aux,
                                                                                                          valid, fake)
                recon_loss_lst.append(recon_loss)

                dis_loss = (d_real_loss.item() + d_fake_loss.item()) / 2
                gen_loss = g_loss.item()

                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_D_x = plot_D_x * smooth_factor + D_x.item() * (1 - smooth_factor)
                plot_D_G_z1 = plot_D_G_z1 * smooth_factor + D_G_z1.item() * (1 - smooth_factor)
                plot_D_G_z2 = plot_D_G_z2 * smooth_factor + D_G_z2.item() * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)

                dis_loss_lst.append(plot_dis_s / plot_ws)
                gen_loss_lst.append(plot_gen_s / plot_ws)

                D_x_lst.append(plot_D_x / plot_ws)
                D_G_z1_lst.append(plot_D_G_z1 / plot_ws)
                D_G_z2_lst.append(plot_D_G_z2 / plot_ws)

                self.global_step += 1

            noisy_input = self.trick_dict.get('noisy_input', None)
            if noisy_input:
                print('Noisy input sigma: {:.4f}'.format(noisy_input['sigma']))

            if checkpoint_path and (epoch + 1) % epoch_per_save == 0:
                gan_model.save_checkpoint(checkpoint_path)

            # plot loss figure
            step = [a for a in range(self.global_step - len(dis_loss_lst), self.global_step)]

            self.plotter.plot('gan_loss', 'dis_loss', step, dis_loss_lst)
            self.plotter.plot('gan_loss', 'gen_loss', step, gen_loss_lst)
            self.plotter.plot('gan_loss', 'recon_loss', step, recon_loss_lst)
            self.plotter.plot('gan_output', 'D_x', step, D_x_lst)
            self.plotter.plot('gan_output', 'D_G_z1', step, D_G_z1_lst)
            self.plotter.plot('gan_output', 'D_G_z2', step, D_G_z2_lst)

            # callbacks
            for callback in callbacks:
                callback(self, gan_model)

        if checkpoint_path:
            gan_model.save_checkpoint(checkpoint_path)
