import numpy as np


def delta(prev):
    # return prev * (1 - prev)
    return np.dot(prev, (1 - prev))
