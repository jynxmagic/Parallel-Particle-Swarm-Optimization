from math import sqrt
from pathlib import Path

import numpy as np

FILENAME = "boston.csv"


def get_coef(x, y):
    r_t = (y.size * np.sum(x * y)) - (np.sum(x) * np.sum(y))
    r_b = (y.size * np.sum(x ** 2) - np.sum(x) ** 2) * (
        y.size * np.sum(y ** 2) - np.sum(y) ** 2
    )
    r_b = sqrt(r_b)
    r = r_t / r_b
    return r


def calculate_line(x, y):
    m = x.size * np.sum(x * y) - np.sum(x) * np.sum(y)
    m = m / (x.size * np.sum(x ** 2) - np.sum(x) ** 2)

    b = np.sum(y) - m * np.sum(x)
    b = b / x.size

    return [m, b]


def boston(x_col, y_col, x_pred):
    data = np.loadtxt(Path.cwd() / FILENAME)

    y = data[..., y_col]
    x = data[:, x_col]

    coef = get_coef(x, y)

    line = calculate_line(x, y)
    print(line)
    y_pred = line[0] * x_pred + line[1]  # y = mx+b

    print(y_pred)

    np.set_printoptions(suppress=True)  # non-scientific notation


boston(5, 13, 7.1)  # 5=No2/o2,13=avgpri
