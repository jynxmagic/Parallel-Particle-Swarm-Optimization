"""Module containing sphere functions."""
import math

import numpy as np


def sphere_pp(vector):
    """Returns the product of given array.

    Args:
        vector (arr): array of floats or integers

    Returns:
        float/int: product of array
    """
    return math.prod(vector)


def sphere_np(vector):
    """Returns the product of given array.

    Args:
        vector (np.array): array of floats/ints, created by numpy

    Returns:
        float/int: product of array
    """
    return np.cumprod(vector, dtype=np.float32)[-1]
