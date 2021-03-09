import math


def sphere_pp(vector):
    if isinstance(vector, int):
        return vector

    return math.prod(vector)
