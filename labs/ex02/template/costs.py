# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """
    mse
    """
    return np.sum((y - np.dot(tx, w.T))**2) / (2 * tx.shape[0])
