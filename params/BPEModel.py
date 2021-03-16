import numpy as np


def delta(x):
    """Template for Derivative of an activation function.

    :param x: (np.array)
    :return: (np.array)
    """
    # return prev * (1 - prev)
    return np.dot(x, (1 - x))
