"""
Utilities for Semi-supervised GAN
"""

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from visdom import Visdom

from lib.common import FloatTensor, LongTensor


def compute_accuracy(data_loader: DataLoader, gan_model):
    gan_model._set_to_eval()
    correct = 0
    total = 0
    for input_data in tqdm(data_loader):
        images = input_data[0].type(FloatTensor)
        labels = input_data[1].type(LongTensor)
        out = gan_model.discriminator(images)[1:]
        for i, label in enumerate(out):
            _, predicted = torch.max(label.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    gan_model._set_to_train()
    return accuracy

class SampleImage(object):
    """ Make a grid and plot a class of image on one row """

    def __init__(self, n_row, nb_classes, code_size):
        self.n_row = n_row
        self.viz = Visdom()
        self.fixed_z = torch.from_numpy(np.random.normal(0, 1, (n_row * nb_classes, code_size))).type(FloatTensor)
        self.win = None

    def __call__(self, trainer, gan_model):
        global_step = trainer.global_step
        output_images = (gan_model.generate('fixed', self.fixed_z) + 1.) / 2.

        if self.win is None:
            self.win = self.viz.images(output_images, nrow=self.n_row, opts=dict(
                caption='Step: {}'.format(global_step)
            ))
        else:
            self.viz.images(output_images, nrow=self.n_row, win=self.win, opts=dict(
                caption='Step: {}'.format(global_step)
            ))
