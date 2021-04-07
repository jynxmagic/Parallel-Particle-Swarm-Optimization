from pathlib import Path

import numpy as np

FILENAME = 'boston.csv'

def boston(x):
    data = np.loadtxt(Path.cwd() / FILENAME)
