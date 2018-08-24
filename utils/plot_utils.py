"""
Plot utilities
"""
import numpy as np
from visdom import Visdom


def viz_grid(Xs, padding=0):
    N, H, W, C = Xs.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size + 1)
    grid_width = W * grid_size + padding * (grid_size + 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = padding, H + padding
    for y in range(grid_size):
        x0, x1 = padding, W + padding
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            if type(x) == list:
                if len(x) == 1:
                    x = [x[0], x[0]]
                    y = [y[0], y[0]]
                plot_X = np.array(x)
                plot_Y = np.array(y)
            elif type(x) == np.ndarray:
                plot_X = x
                plot_Y = y
            else:
                plot_X = np.array([x, x])
                plot_Y = np.array([y, y])
            self.plots[var_name] = self.viz.line(X=plot_X, Y=plot_Y, env=self.env, opts=dict(
                legend=split_name,
                showlegend=True,
                title=var_name,
                xlabel='Steps',
                ylabel=var_name
            ))
        else:
            if type(x) == list:
                plot_X = np.array(x)
                plot_Y = np.array(y)
            elif type(x) == np.ndarray:
                plot_X = x
                plot_Y = y
            else:
                plot_X = np.array([x])
                plot_Y = np.array([y])
            self.viz.line(X=plot_X, Y=plot_Y, env=self.env, win=self.plots[var_name], update='append',
                          opts=dict(legend=split_name))


visdom_line_plotter = {}


def get_visdom_line_plotter(env_name) -> VisdomLinePlotter:
    if env_name not in visdom_line_plotter:
        visdom_line_plotter[env_name] = VisdomLinePlotter(env_name)
    return visdom_line_plotter[env_name]


def unittest():
    x = np.arange(0, 100)
    a = np.random.randn(100, 3)
    split_name = ['a', 'b', 'c']
    plotter = get_visdom_line_plotter('main')
    plotter.plot('test', split_name, x=x, y=a)
    x = np.arange(100, 300)
    a = np.random.randn(200, 3)
    plotter.plot('test', split_name, x=x, y=a)
