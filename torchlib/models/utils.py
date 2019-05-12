import torch
import torch.nn as nn

from torchlib.common import map_location


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str) -> None:
    model.load_state_dict(torch.load(path, map_location=map_location))
