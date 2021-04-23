import pprint
from math import sqrt
from pathlib import Path

import numpy as np
from matplotlib import pyplot

FILENAME = "boston.csv"
TARGET_COLUMN = 13


def generate_predictions(data, m_factors, b0):
    row_count, col_count = data.shape
    col_count -= 1  # last row is y

    # calculate predictions for each row
    pred_rows = []
    for row in range(row_count):
        y_pred_for_row = b0  # y = b0
        for index in range(col_count):
            m = m_factors[index]
            y_pred_for_row += m * data[row][index]  # y += sum(mx)
        pred_rows.append(y_pred_for_row)
    return pred_rows

def rmse(y_preds, y_actuals):
    row_count = len(y_actuals)
    # calculate rmse
    i = 0
    sum_of_square_err = 0
    for pred in y_preds:
        actual = y_actuals[i]
        diff = actual - pred
        sum_of_square_err += diff ** 2
        i = i + 1
    sum_of_square_err = sum_of_square_err / row_count
    sum_of_square_err = sqrt(sum_of_square_err)
    return sum_of_square_err

def show_scatter(y, pred_rows):
    pyplot.scatter(y, pred_rows)
    y_lim = pyplot.ylim()
    x_lim = pyplot.xlim()
    pyplot.plot(x_lim, y_lim, 'k-', color='r')
    pyplot.ylim(y_lim)
    pyplot.xlim(x_lim)
    pyplot.show()

def boston(z):
    data = np.loadtxt(Path.cwd() / FILENAME)

    target = TARGET_COLUMN
    y = data[..., target]

    b0 = z[len(z) - 1]  # last item is intersect from particle swarm

    predictions = generate_predictions(data, z, b0)
    root_sqaured_error = rmse(predictions, y)
    show_scatter(y,predictions)

    return root_sqaured_error


print(
    boston(
        [
0.31508836290904185,   0.10215494010464843 ,  0.17640535181769604,
 -0.3389466361246862  ,  0.214185500169508   ,  0.2541842545786732,
  0.017954290537926687 ,-0.2893618089742949 ,   0.38518203275115304,
 -0.011031121093775027 , 0.11782033008772752 ,  0.05271352960511161,
 -0.39711665111370104  , 0.48755429311700427
        ]
    )
)
