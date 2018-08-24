"""
Trainer class for WGAN
"""

import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from lib.common import FloatTensor
from lib.utils.plot_utils import get_visdom_line_plotter


class Trainer(object):
    def __init__(self, num_iter_D=5, clip=0.01, trick_dict=None):
        if trick_dict is None:
            self.trick_dict = {}
        else:
            self.trick_dict = trick_dict
        self.global_step = 0
        self.plotter = get_visdom_line_plotter('main')
        self.num_iter_D = num_iter_D
        self.clip = clip

    def _create_real_data(self, raw_real_data):
        noisy_input = self.trick_dict.get('noisy_input', None)
        if noisy_input:
            raw_real_data = raw_real_data + torch.from_numpy(
                np.random.randn(*raw_real_data.shape) * noisy_input['sigma']).type(torch.FloatTensor)
            noisy_input['sigma'] = max(0, noisy_input['sigma'] - noisy_input['decay'])
        real_data = Variable(raw_real_data.type(FloatTensor))
        return real_data

    def train(self, num_epoch, data_loader, gan_model, checkpoint_path, epoch_per_save, callbacks):

        for epoch in range(num_epoch):
            # we sample a batch after each epoch
            dis_loss_lst = []
            gen_loss_lst = []
            D_x_lst = []
            D_G_z1_lst = []
            D_G_z2_lst = []
            # plot smoothing
            smooth_factor = 0.95
            plot_s = 0
            plot_D_x = 0
            plot_D_G_z1 = 0
            plot_D_G_z2 = 0
            plot_ws = 0

            d_real_loss, D_x, D_G_z1 = 0, 0, 0

            print('Epoch {}'.format(epoch + 1))
            for i, input_and_aux in enumerate(tqdm(data_loader)):
                # We assume the input_and_label is a tuple containing data and auxiliary information

                batch_size = input_and_aux[0].shape[0]
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, gan_model.code_size))))
                if (i + 1) % self.num_iter_D != 0:
                    # train dis
                    real_data = self._create_real_data(input_and_aux[0])
                    d_real_loss, D_x, D_G_z1 = gan_model._train_dis(real_data, z, self.clip)

                else:
                    g_loss, D_G_z2 = gan_model._train_gen(z)
                    gen_loss = g_loss.item()

                    dis_loss = d_real_loss.item()
                    plot_dis_s = plot_s * smooth_factor + dis_loss * (1 - smooth_factor)
                    plot_D_x = plot_D_x * smooth_factor + D_x.item() * (1 - smooth_factor)
                    plot_D_G_z1 = plot_D_G_z1 * smooth_factor + D_G_z1.item() * (1 - smooth_factor)
                    plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)

                    dis_loss_lst.append(plot_dis_s / plot_ws)
                    D_x_lst.append(plot_D_x / plot_ws)
                    D_G_z1_lst.append(plot_D_G_z1 / plot_ws)

                    plot_gen_s = plot_s * smooth_factor + gen_loss * (1 - smooth_factor)
                    plot_D_G_z2 = plot_D_G_z2 * smooth_factor + D_G_z2.item() * (1 - smooth_factor)

                    gen_loss_lst.append(plot_gen_s / plot_ws)
                    D_G_z2_lst.append(plot_D_G_z2 / plot_ws)
                    self.global_step += 1

            noisy_input = self.trick_dict.get('noisy_input', None)
            if noisy_input:
                print('Noisy input sigma: {:.4f}'.format(noisy_input['sigma']))

            if checkpoint_path and (epoch + 1) % epoch_per_save == 0:
                gan_model.save_checkpoint(checkpoint_path)

            # plot loss figure
            step = [a for a in range(self.global_step - len(gen_loss_lst), self.global_step)]

            self.plotter.plot('gan_loss', 'dis_loss', step, dis_loss_lst)
            self.plotter.plot('gan_loss', 'gen_loss', step, gen_loss_lst)
            self.plotter.plot('gan_output', 'D_x', step, D_x_lst)
            self.plotter.plot('gan_output', 'D_G_z1', step, D_G_z1_lst)
            self.plotter.plot('gan_output', 'D_G_z2', step, D_G_z2_lst)

            # callbacks
            for callback in callbacks:
                callback(self, gan_model)

        if checkpoint_path:
            gan_model.save_checkpoint(checkpoint_path)
