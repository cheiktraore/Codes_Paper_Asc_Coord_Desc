import numpy as np
from numpy.linalg import norm
from numba import njit

@njit
def ST(x, u):
    """Soft-thresholding of vector x at level u, i.e., entrywise:
    x_i + u_i if x_i < -u_i, x_i - u_i if x_i > u_i and 0 else.
    """
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


@njit
def lasso_loss(A, b, lbda, x):
    """Value of Lasso objective at x."""
    return norm(A @ x - b) ** 2 / 2. + lbda * norm(x, ord=1)
