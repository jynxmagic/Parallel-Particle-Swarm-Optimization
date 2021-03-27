"""Module containing sphere functions."""
import numpy as np


def sphere_np(vector):
    """Returns the sphere function of a vector.

    Args:
        vector (np.array): array of floats/ints, created by numpy

    Returns:
        float/int: sphere function of array
    """

    return np.sum(vector ** 2)
