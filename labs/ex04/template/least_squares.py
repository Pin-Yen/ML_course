# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    A = np.dot(tx.T, tx)
    B = np.dot(tx.T, y)
    # Aw = B
    weights = np.linalg.solve(A, B)
    return weights

