"""
Trainer for BiGAN/ALI
"""

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from ....common import FloatTensor
from ....utils.plot import get_visdom_line_plotter


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
        real_data = raw_real_data.type(FloatTensor)
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

    def train(self, num_epoch, data_loader, gan_model, checkpoint_path, epoch_per_save, callbacks):

        for epoch in range(num_epoch):
            # we sample a batch after each epoch
            dis_loss_lst = []
            gen_loss_lst = []
            D_x_lst = []
            D_G_z_lst = []
            # plot smoothing
            smooth_factor = 0.95
            plot_dis_s = 0
            plot_gen_s = 0
            plot_D_x = 0
            plot_D_G_z = 0
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

                # sample noise
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, gan_model.code_size))))
                real_data = self._create_real_data(input_and_aux[0])

                d_loss, g_loss, D_x, D_G_z = gan_model._train(real_data, z, valid, fake)

                dis_loss = d_loss.item()
                gen_loss = g_loss.item()

                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_D_x = plot_D_x * smooth_factor + D_x.item() * (1 - smooth_factor)
                plot_D_G_z = plot_D_G_z * smooth_factor + D_G_z.item() * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)

                dis_loss_lst.append(plot_dis_s / plot_ws)
                gen_loss_lst.append(plot_gen_s / plot_ws)

                D_x_lst.append(plot_D_x / plot_ws)
                D_G_z_lst.append(plot_D_G_z / plot_ws)

                self.global_step += 1

            noisy_input = self.trick_dict.get('noisy_input', None)
            if noisy_input:
                print('Noisy input sigma: {:.4f}'.format(noisy_input['sigma']))

            if checkpoint_path and (epoch + 1) % epoch_per_save == 0:
                gan_model.save_checkpoint(checkpoint_path)

            # plot loss figure
            step = [a for a in range(self.global_step - len(dis_loss_lst), self.global_step)]

            data = np.array([dis_loss_lst, gen_loss_lst]).transpose()
            legend = ['dis_loss', 'gen_loss']
            self.plotter.plot('gan_loss', legend, step, data)
            data = np.array([D_x_lst, D_G_z_lst]).transpose()
            legend = ['D_x', 'D_G_z']
            self.plotter.plot('gan_output', legend, step, data)

            # callbacks
            for callback in callbacks:
                callback(self, gan_model)

        if checkpoint_path:
            gan_model.save_checkpoint(checkpoint_path)
