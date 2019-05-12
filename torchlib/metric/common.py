"""
Define common metrics for measuring performance.
"""

import numpy as np
import sklearn


def accuracy_score(y_score: np.ndarray, y_pred: np.ndarray) -> float:
    """ Compute accuracy score. Usually followed by softmax to compute probability.

    Args:
        y_score: numpy array of shape (batch_size, C, d1, d2, ..., dk)
        y_pred: (batch_size, d1, d2, ..., dk)

    Returns:

    """
    y_true = np.argmax(y_score, axis=1)
    assert y_true.shape == y_pred.shape, 'y_true and y_pred must have the same shape. Got y_true {}, y_pred {}'.format(
        y_true.shape, y_pred.shape)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def binary_accuracy_score(y_score: np.ndarray, y_pred: np.ndarray) -> float:
    """ Compute binary accuracy that output from sigmoid.

    Args:
        y_score: (batch_size, d1, d2, ..., dk)
        y_pred: (batch_size, d1, d2, ..., dk)

    Returns:

    """
    y_true = (y_score > 0.5).astype(np.int)
    assert y_true.shape == y_pred.shape, 'y_true and y_pred must have the same shape. Got y_true {}, y_pred {}'.format(
        y_true.shape, y_pred.shape)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
