"""
A submodule that implements models in torchlib using Tensorflow 2.0
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

__all__ = ['tf']
