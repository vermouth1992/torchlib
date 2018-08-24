import numpy as np
from visdom import Visdom


class SampleImage(object):
    """ Make a grid and plot a class of image on one row """

    def __init__(self, n_row, code_size):
        self.n_row = n_row
        self.viz = Visdom()
        self.fixed_z = np.random.normal(0, 1, (n_row * n_row, code_size))
        self.win = None

    def __call__(self, trainer, gan_model):
        if trainer:
            global_step = trainer.global_step
        else:
            global_step = 0
        output_images = (gan_model.generate('fixed', self.fixed_z) + 1.) / 2.

        if self.win is None:
            self.win = self.viz.images(output_images, nrow=self.n_row, opts=dict(
                caption='Step: {}'.format(global_step)
            ))
        else:
            self.viz.images(output_images, nrow=self.n_row, win=self.win, opts=dict(
                caption='Step: {}'.format(global_step)
            ))