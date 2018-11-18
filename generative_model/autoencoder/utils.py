"""
Implement a list of callbacks
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter
from torchlib.common import FloatTensor


class SampleImage(object):
    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.fixed_z = None

    def __call__(self, epoch, model, summary_writer: SummaryWriter):
        if self.fixed_z is None:
            self.fixed_z = model.sample_latent_code(self.n_row * self.n_col)

        image = model.decode(self.fixed_z)
        x = torchvision.utils.make_grid(image, nrow=self.n_row)
        summary_writer.add_image('fig/Samples', x, epoch)


class Reconstruction(object):
    def __init__(self, images):
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        self.images = images
        self.nrow = int(np.sqrt(self.images.shape[0]))
        self.initialized = False

    def __call__(self, n_iter, model, summary_writer: SummaryWriter):
        if not self.initialized:
            summary_writer.add_image('fig/Original', torchvision.utils.make_grid(self.images, nrow=self.nrow), 0)
            self.initialized = True

        reconstructed_images = model.reconstruct(self.images.type(FloatTensor))
        summary_writer.add_image('fig/Reconstructed', torchvision.utils.make_grid(reconstructed_images,
                                                                              nrow=self.nrow), n_iter)


class VisualizeLatent(object):
    def __init__(self, data_loader, method='tsne'):
        self.data_loader = data_loader
        self.labels = []
        for _, label in self.data_loader:
            self.labels.append(label)
        self.labels = torch.cat(self.labels, 0).cpu().numpy() + 1
        if method == 'tsne':
            self.model = TSNE
        elif method == 'pca':
            self.model = PCA

    def __call__(self, n_iter, model, summary_writer: SummaryWriter):
        latent_list = []
        for data, _ in self.data_loader:
            latent = model.encode_reparm(data.type(FloatTensor))
            latent_list.append(latent)
        latent = torch.cat(latent_list, 0).detach().cpu().numpy()
        if latent.shape[1] > 2:
            latent = self.model(n_components=2).fit_transform(latent)

        fig = plt.figure()
        plt.scatter(latent[:, 0], latent[:, 1], c=self.labels)
        summary_writer.add_figure('fig/Latent', fig, global_step=n_iter)
