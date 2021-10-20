# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from batch_iter import batch_iter
from costs import compute_loss
from proj1_helpers import predict_labels

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - np.dot(tx, w.T)

    return -(np.dot(tx.T, e))/y.shape[0]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
 
    ws = [initial_w]
    losses = []
    accurarcies = []

    w = initial_w
    for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        loss = compute_loss(batch_y, batch_tx, w)
        grad = compute_stoch_gradient(batch_y, batch_tx, w)
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        accurarcies.append((batch_y==predict_labels(w, batch_tx)).mean())

    return losses, accurarcies, ws