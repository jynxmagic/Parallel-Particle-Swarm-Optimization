import pprint
from math import sqrt
from pathlib import Path

import numpy as np

FILENAME = "boston.csv"
TARGET_COLUMN = 13


def get_coef(x, y):
    # pearsons coef
    r_t = (y.size * np.sum(x * y)) - (np.sum(x) * np.sum(y))
    r_b = (y.size * np.sum(x ** 2) - np.sum(x) ** 2) * (
        y.size * np.sum(y ** 2) - np.sum(y) ** 2
    )
    r_b = sqrt(r_b)
    r = r_t / r_b
    return r


def calculate_line(x, y):
    m = x.size * np.sum(x * y) - np.sum(x) * np.sum(y)  # slope
    m = m / (x.size * np.sum(x ** 2) - np.sum(x) ** 2)

    b = np.sum(y) - m * np.sum(x)  # y-intercept
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

    target = TARGET_COLUMN
    y = data[..., target]

    row_count = data.shape[0]
    col_count = data.shape[1] - 1  # last row is y

    b0 = z[len(z) - 1]  # last item is intersect

    # calculate predictions for each row
    pred_rows = []
    for row in range(row_count):
        y_pred_for_row = b0  # y = b0
        for index in range(col_count):
            m = z[index]
            y_pred_for_row += m * data[row][index]  # y += sum(mx)
        pred_rows.append(y_pred_for_row)

    # calculate rmse
    i = 0
    sum_of_square_err = 0
    for pred in pred_rows:
        actual = y[i]
        print(i, pred, actual)
        diff = actual - pred
        sum_of_square_err += diff ** 2
        i = i + 1

    sum_of_square_err = sum_of_square_err / row_count
    sum_of_square_err = sqrt(sum_of_square_err)

    return sum_of_square_err


print(
    boston(
        [
            0.08874423755159883,
            0.11859090070123664,
            0.13421402085885736,
            0.17910779839675817,
            0.06258282505018851,
            0.1397721230917058,
            -0.0043082689434887685,
            0.15128850247700576,
            0.09102509189748319,
            -0.0053155251219678045,
            0.028307605305521486,
            0.042917146650710156,
            0.072188950363832,
            -0.003588613959859436,
        ]
    )
)
