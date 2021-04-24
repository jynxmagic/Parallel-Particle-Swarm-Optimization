from math import sqrt
from pathlib import Path

import numpy as np
from matplotlib import pyplot

FILENAME = "boston.csv"
TARGET_COLUMN = 13


def generate_predictions(data, m_factors, b0):
    row_count, col_count = data.shape
    col_count -= 1  # last col is y

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
    sum_of_square_err = 0
    for index, pred in enumerate(y_preds):
        actual = y_actuals[index]
        diff = actual - pred
        sum_of_square_err += diff ** 2
    sum_of_square_err = sum_of_square_err / row_count
    sum_of_square_err = sqrt(sum_of_square_err)
    return sum_of_square_err

def show_scatter(y, pred_rows):
    pyplot.scatter(y, pred_rows)
    pyplot.show()

def boston(particles):
    data = np.loadtxt(Path(__file__).parent.absolute() / FILENAME)

    target = TARGET_COLUMN
    y = data[..., target]

    b0 = particles[len(particles) - 1]  # last item is intersect from particle swarm

    predictions = generate_predictions(data, particles, b0)
    root_sqaured_error = rmse(predictions, y)
    #show_scatter(y,predictions)

    return root_sqaured_error


print(
    boston(
        [
-0.08486625408953588,   0.1076806851258205 ,  -0.34254048462604053,
 -0.05397952819161085,  -0.264129182712456,     0.17680983829948105,
  0.09105753747310638,  -0.049446094084896586,  0.08843132271793035,
  0.009218846082991602,  0.17882554165928058,   0.04765998014759277,
 -0.5279020901957536,   -0.27173104098166445,
        ]
    )
)
