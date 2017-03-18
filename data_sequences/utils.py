import os

import numpy as np


def makedirs(f_name):
    """same as os.makedirs(f_name, exists_ok=True) at python3"""
    if not os.path.exists(f_name):
        os.makedirs(f_name)


def nd_array_to_one_hot(arr, n_classes):
    """Convert last dimmension of numpy ND array to one hot representation
    Args:
        arr: ND numpy array.
        n_classes: `int`.
    Return:
        converted: numpy array with shape [*ND.shape, n_classes] on hot encoded
    """
    initial_shape = list(arr.shape)
    arr = arr.reshape(-1)
    converted = np.zeros((arr.shape[0], n_classes), dtype=arr.dtype)
    converted[range(arr.shape[0]), arr] = np.ones(arr.shape, dtype=arr.dtype)
    initial_shape.append(n_classes)
    converted = converted.reshape(initial_shape)
    return converted


def nd_array_from_one_hot(arr):
    """Convert ND one hot numpy array to N-1D numpy array.
    Args:
        arr: ND numpy array one hot encoded
    Return:
        converted: N-1D numpy array
    """
    converted = np.argmax(arr, axis=len(arr.shape) - 1)
    return converted
