"""

The optimization test problems defined below are numpy/python implementations
of the mathmatical formulae, solutions, and dimensions defined in this
link: https://www.sfu.ca/~ssurjano/optimization.html
"""
import numpy as np


def rosenbrock(x):
    """
    x ∈ [-5, 10]
    global minima score = 0
    global minima position = [1, ..., 1]
    """
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def sphere_np(x):
    """
    x ∈ [-5.12, 5.12]
    global minima score = 0
    global minima position = [0, ..., 0]
    """
    return np.sum(x ** 2)


def griewank(x):
    """
    x ∈ [-600, 600]
    global minima score = 0
    global minima position = [0, ..., 0]
    """
    return (
        np.sum(x ** 2 / 4000)
        - np.product(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))
        + 1
    )


def zacharov(x):
    """
    x ∈ [-5, 10]
    global minima score = 0
    global minima position = [0, ..., 0]
    """
    return (
        np.sum(x ** 2)
        + (np.sum(0.5 * np.arange(1, x.size + 1) * x)) ** 2
        + (np.sum(0.5 * np.arange(1, x.size + 1) * x)) ** 4
    )
