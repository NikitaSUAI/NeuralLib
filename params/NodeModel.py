import numpy as np
from scipy.special import expit


def my_sigmoid(x):
    """ Template of an activation function. This template better with small
    arrays.

    :param x: (np.array)
    :return: (np.array)

    """
    return 1/(1 + np.exp(-x))


def sigmoid(x):
    """ Template of an activation function. This template better with large
    arrays. It call scipy.special.expit().

    :param x: (np.array)
    :return: (np.array)
    """
    return expit(x)
