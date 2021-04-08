from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FILENAME = "boston.csv"


def boston(z):
    data = np.loadtxt(Path.cwd() / FILENAME)

    y = data[..., 13]
    x = data[:, 5]

    model = np.poly1d(np.polyfit(x, y, 3))
    line = np.linspace(np.min(x), np.max(x))

    print("x", x)
    print("y", y)

    plt.scatter(x, y)
    plt.plot(line, model(line))
    plt.show()
    np.set_printoptions(suppress=True)  # non-scientific notation


boston(1)
