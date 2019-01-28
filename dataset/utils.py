import torch
from torch.utils.data import TensorDataset
from torchlib.common import enable_cuda


def create_tensor_dataset(data):
    tensor_data = [torch.from_numpy(d) for d in data]
    return TensorDataset(*tensor_data)


def create_data_loader(data, batch_size=32):
    """ Create a data loader given numpy array x and y

    Args:
        data: a tuple (x, y, z, ...) where they have common first shape dim.

    Returns: Pytorch data loader

    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if enable_cuda else {}
    dataset = create_tensor_dataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, **kwargs)
    return loader
