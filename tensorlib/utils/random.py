import random

import numpy as np
import tensorflow as tf


def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
