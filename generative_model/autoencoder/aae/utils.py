"""
Utilities for AAE
"""

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from visdom import Visdom

from ....common import FloatTensor

class SampleImage(object):
    """ Make a grid and plot a class of image on one row """

    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.viz = Visdom()
        self.fixed_z = None
        self.win = None

    def __call__(self, trainer, gan_model):
        if trainer:
            global_step = trainer.global_step
        else:
            global_step = 0
        if self.fixed_z is None:
            self.fixed_z = gan_model.sample_latent_code(self.n_row * self.n_col)

        output_images = (gan_model.generate('fixed', self.fixed_z) + 1.) / 2.

        if self.win is None:
            self.win = self.viz.images(output_images, nrow=self.n_row, opts=dict(
                caption='Step: {}'.format(global_step)
            ))
        else:
            self.viz.images(output_images, nrow=self.n_row, win=self.win, opts=dict(
                caption='Step: {}'.format(global_step)
            ))


class Reconstruction(object):
    def __init__(self, data_loader):
        self.images, _ = next(iter(data_loader))
        self.images = self.images.type(FloatTensor)
        assert self.images.shape[0] == 100
        self.viz = Visdom()
        self.win_original = None
        self.win_recon = None

    def __call__(self, trainer, gan_model):
        recon = gan_model.reconstruct(self.images)
        if self.win_original is None:
            self.win_original = self.viz.images(self.images, nrow=10, opts=dict(
                caption='Original Images'
            ))
        if self.win_recon is None:
            self.win_recon = self.viz.images(recon, nrow=10, opts=dict(
                caption='Reconstructed Images'
            ))
        else:
            self.viz.images(recon, nrow=10, win=self.win_recon, opts=dict(
                caption='Reconstructed Images'
            ))


class VisualizeLatent(object):
    def __init__(self, data_loader, method='tsne'):
        self.data_loader = data_loader
        self.labels = []
        for _, label in self.data_loader:
            self.labels.append(label)
        self.labels = torch.cat(self.labels, 0).cpu().numpy() + 1
        self.viz = Visdom()
        self.win = None
        if method == 'tsne':
            self.model = TSNE
        elif method == 'pca':
            self.model = PCA
        else:
            raise ValueError('Unknown embedding method')

    def __call__(self, trainer, gan_model):
        latent_list = []
        for data, _ in self.data_loader:
            latent = gan_model.encoder(data.type(FloatTensor))
            latent_list.append(latent)
        latent = torch.cat(latent_list, 0).detach().cpu().numpy()
        if latent.shape[1] > 2:
            latent = self.model(n_components=2).fit_transform(latent)
        if self.win is None:
            self.win = self.viz.scatter(latent, self.labels, opts=dict(
                markersize=5
            ))
        else:
            self.viz.scatter(latent, self.labels, win=self.win)
