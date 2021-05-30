import numpy as np


def sort_xy(x, y):
    sorter = np.argsort(x)
    return x[sorter], y[sorter]
