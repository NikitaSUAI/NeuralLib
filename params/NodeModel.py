import numpy as np
from scipy.special import expit


def my_sigmoid(x):
    # better with small arrays and single value
    return 1/(1 + np.exp(-x))


def sigmoid(x):
    # better with large arrays
    return expit(x)


if __name__ == "__main__":
    print(my_sigmoid(0.3 * my_sigmoid(0.7 * 0.8)))