# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.reshape(x, (-1, 1))
    return np.hstack([x**j for j in range(0, degree+1)])
