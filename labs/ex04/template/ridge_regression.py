# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_pron = 2 * y.shape[0] * lambda_
    A = np.dot(tx.T, tx) + lambda_pron * np.identity(tx.shape[1])
    B = np.dot(tx.T, y)
    # Aw = B
    weights = np.linalg.solve(A, B)

    return weights
