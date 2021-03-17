import numpy


def sphere_pp(vector):
    tot = 0

    for value in vector:
        tot += value * value

    return tot


def sphere_np(array):
    return numpy.cumprod(array, dtype=float)[-1]
