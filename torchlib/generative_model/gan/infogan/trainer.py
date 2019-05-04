"""
Trainer for infogan
"""

import numpy as np
import torch
from torchlib.common import FloatTensor
from torchlib.utils.plot import get_visdom_line_plotter
from torch.autograd import Variable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, trick_dict=None):
        if trick_dict is None:
            self.trick_dict = {}
        else:
            self.trick_dict = trick_dict
        self.global_step = 0
        self.plotter = get_visdom_line_plotter('main')

    def _create_real_data(self, raw_real_data):
        noisy_input = self.trick_dict.get('noisy_input', None)
        if noisy_input:
            raw_real_data = raw_real_data + torch.from_numpy(
                np.random.randn(*raw_real_data.shape) * noisy_input['sigma']).type(torch.FloatTensor)
            noisy_input['sigma'] = max(0, noisy_input['sigma'] - noisy_input['decay'])
        real_data = Variable(raw_real_data.type(FloatTensor))
        return real_data

    def _create_valid(self, batch_size):
        soft_label = self.trick_dict.get('label_smooth', None)
        if soft_label:
            valid_range = soft_label['valid_range']
        else:
            valid_range = 1.
        if isinstance(valid_range, list):
            valid = Variable(FloatTensor(batch_size, 1).uniform_(*valid_range), requires_grad=False)
        else:
            valid = Variable(FloatTensor(batch_size, 1).fill_(valid_range), requires_grad=False)
        return valid

    def _create_fake(self, batch_size):
        soft_label = self.trick_dict.get('label_smooth', None)
        if soft_label:
            fake_range = soft_label['fake_range']
        else:
            fake_range = 0.
        if isinstance(fake_range, list):
            fake = Variable(FloatTensor(batch_size, 1).uniform_(*fake_range), requires_grad=False)
        else:
            fake = Variable(FloatTensor(batch_size, 1).fill_(fake_range), requires_grad=False)
        return fake

    def train(self, num_epoch, data_loader, gan_model, disc_iter, checkpoint_path, epoch_per_save, callbacks):
        assert disc_iter > 0, 'Discriminator update iteration must be greater than zero'
        for epoch in range(num_epoch):
            gan_model._set_to_train()
            dis_loss_lst = []
            gen_loss_lst = []
            info_loss_lst = []
            D_x_lst = []
            D_G_z1_lst = []
            D_G_z2_lst = []
            # plot smoothing
            smooth_factor = 0.95
            plot_dis_s = 0
            plot_gen_s = 0
            plot_info_s = 0
            plot_D_x = 0
            plot_D_G_z1 = 0
            plot_D_G_z2 = 0
            plot_ws = 0

            print('Epoch {}'.format(epoch + 1))
            for input_and_aux in tqdm(data_loader):
                # We assume the input_and_label is a tuple containing data and auxiliary information

                # Adversarial ground truths
                batch_size = input_and_aux[0].shape[0]

                valid = self._create_valid(batch_size)
                fake = self._create_fake(batch_size)

                flip_label = self.trick_dict.get('flip_label', None)
                if flip_label and (self.global_step + 1) % flip_label['num_steps_per_flip'] == 0:
                    valid, fake = fake, valid

                # train discriminator
                real_data = self._create_real_data(input_and_aux[0])
                for _ in range(disc_iter):
                    d_real_loss, D_x = gan_model._train_dis_with_real(real_data, valid)
                    d_fake_loss, D_G_z1 = gan_model._train_dis_with_fake(fake)

                # train generator
                valid = self._create_valid(2 * batch_size)
                g_loss, D_G_z2 = gan_model._train_gen(valid)

                i_loss = gan_model._train_info(batch_size)

                # gan_model.update_parameters()
                dis_loss = (d_real_loss.item() + d_fake_loss.item()) / 2
                gen_loss = g_loss.item()
                info_loss = i_loss.item()

                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_info_s = plot_info_s * smooth_factor + info_loss * (1 - smooth_factor)
                plot_D_x = plot_D_x * smooth_factor + D_x.item() * (1 - smooth_factor)
                plot_D_G_z1 = plot_D_G_z1 * smooth_factor + D_G_z1.item() * (1 - smooth_factor)
                plot_D_G_z2 = plot_D_G_z2 * smooth_factor + D_G_z2.item() * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)

                dis_loss_lst.append(plot_dis_s / plot_ws)
                gen_loss_lst.append(plot_gen_s / plot_ws)
                info_loss_lst.append(plot_info_s / plot_ws)

                D_x_lst.append(plot_D_x / plot_ws)
                D_G_z1_lst.append(plot_D_G_z1 / plot_ws)
                D_G_z2_lst.append(plot_D_G_z2 / plot_ws)

                self.global_step += 1

            if gan_model.optimizer_G_scheduler:
                gan_model.optimizer_G_scheduler.step()
            if gan_model.optimizer_D_scheduler:
                gan_model.optimizer_D_scheduler.step()

            noisy_input = self.trick_dict.get('noisy_input', None)
            if noisy_input:
                print('Noisy input sigma: {:.4f}'.format(noisy_input['sigma']))

            if checkpoint_path and (epoch + 1) % epoch_per_save == 0:
                gan_model.save_checkpoint(checkpoint_path)

            # plot loss figure
            step = [a for a in range(self.global_step - len(dis_loss_lst), self.global_step)]

            data = np.array([dis_loss_lst, gen_loss_lst, info_loss_lst]).transpose()
            legend = ['dis_loss', 'gen_loss', 'info_loss']
            self.plotter.plot('gan_loss', legend, step, data)
            data = np.array([D_x_lst, D_G_z1_lst, D_G_z2_lst]).transpose()
            legend = ['D_x', 'D_G_z1', 'D_G_z2']
            self.plotter.plot('gan_output', legend, step, data)

            # callbacks
            for callback in callbacks:
                callback(self, gan_model)

        if checkpoint_path:
            gan_model.save_checkpoint(checkpoint_path)
