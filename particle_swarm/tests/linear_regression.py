from math import sqrt
from pathlib import Path

import numpy as np

FILENAME = "boston.csv"

def get_coef(x,y):
    r_t = (y.size * np.sum(x * y)) - (np.sum(x) * np.sum(y))
    r_b = (y.size * np.sum(x ** 2) - np.sum(x) ** 2) * (
        y.size * np.sum(y ** 2) - np.sum(y) ** 2
    )
    r_b = sqrt(r_b)
    r = r_t / r_b
    return r

def boston(x_col, y_col, x_pred):
    data = np.loadtxt(Path.cwd() / FILENAME)

    y = data[..., y_col]
    x = data[:, x_col]

    coef = get_coef(x,y)

    print(coef)

    np.set_printoptions(suppress=True)  # non-scientific notation


boston(5, 13, 6.4) # 5=No2/o2,13=avgpri
