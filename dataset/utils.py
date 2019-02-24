import torch
from torch.utils.data import TensorDataset
from torchlib.common import enable_cuda
import numpy as np

def create_tensor_dataset(data):
    tensor_data = []
    for d in data:
        if isinstance(d, np.ndarray):
            tensor_data.append(torch.from_numpy(d))
        elif isinstance(d, torch.Tensor):
            tensor_data.append(d)
        else:
            raise ValueError('Unknown data type {}'.format(type(d)))
    return TensorDataset(*tensor_data)


def create_data_loader(data, batch_size=32, shuffle=True, drop_last=False):
    """ Create a data loader given numpy array x and y

    Args:
        data: a tuple (x, y, z, ...) where they have common first shape dim.

    Returns: Pytorch data loader

    """
    kwargs = {'num_workers': 0, 'pin_memory': True} if enable_cuda else {}
    dataset = create_tensor_dataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
    return loader
