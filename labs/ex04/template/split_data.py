# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np
import math


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    rand_permutation = np.random.permutation(x.shape[0])
    shuffled_x = x[rand_permutation]
    shuffled_y = y[rand_permutation]

    train_size = math.floor(x.shape[0] * ratio)
    x_train = shuffled_x[:train_size]
    x_validate = shuffled_x[train_size:]
    y_train = shuffled_y[:train_size]
    y_validate = shuffled_y[train_size:]

    return x_train, x_validate, y_train, y_validate