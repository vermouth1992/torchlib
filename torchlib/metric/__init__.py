from typing import Callable

import numpy as np

from .common import accuracy_score, binary_accuracy_score
from .mmd import mmd_loss, maximum_mean_discrepancy

metric_name_to_func = {
    'accuracy': accuracy_score,
    'binary_accuracy': binary_accuracy_score
}


def contains_metric(metric_name: str) -> bool:
    return metric_name in metric_name_to_func


def get_metric_func(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    return metric_name_to_func.get(metric_name, None)
