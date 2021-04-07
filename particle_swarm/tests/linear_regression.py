from math import sqrt
from pathlib import Path

import numpy as np

FILENAME = "boston.csv"


def boston(z):
    data = np.loadtxt(Path.cwd() / FILENAME)


    coef = np.corrcoef(data)
    print(coef)
    np.set_printoptions(suppress=True)  # non-scientific notation

boston(1)
