from math import sqrt
from pathlib import Path

import numpy as np

FILENAME = "boston.csv"


def get_coef(x, y):
    #pearsons coef
    r_t = (y.size * np.sum(x * y)) - (np.sum(x) * np.sum(y))
    r_b = (y.size * np.sum(x ** 2) - np.sum(x) ** 2) * (
        y.size * np.sum(y ** 2) - np.sum(y) ** 2
    )
    r_b = sqrt(r_b)
    r = r_t / r_b
    return r


def calculate_line(x, y):
    m = x.size * np.sum(x * y) - np.sum(x) * np.sum(y) #slope
    m = m / (x.size * np.sum(x ** 2) - np.sum(x) ** 2)

    b = np.sum(y) - m * np.sum(x) # y-intercept
    b = b / x.size

    return [m, b]


def mse(line, x, y):
    size = x.size
    diff = 0
    for index in range(size):
        y_actual = y[index]
        x_actual = x[index]

        y_pred = line[0] * x_actual + line[1]  # y = mx+b

        diff += (y_actual - y_pred) ** 2

    return diff / size


def boston(z):
    data = np.loadtxt(Path.cwd() / FILENAME)

    y = data[..., 13]
    x = data[:, 5]

    coef = get_coef(x, y)

    line = calculate_line(x, y)

    line[0] = z # particle swarm ovveride

    mean_err = mse(line,x,y)

    return mean_err

    np.set_printoptions(suppress=True)  # non-scientific notation
