import numpy as np


def sphere_pp(vector):
    tot = 0

    for value in vector:
        tot += value * value

    return tot

def sphere_np(vector):
    return np.cumprod(vector)[-1]
