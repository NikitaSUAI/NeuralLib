import numpy as np


def delta(err, prev, next):
    return np.dot(err * prev * (1 - prev), next.T)