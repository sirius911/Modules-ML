import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
    This function should """

    # control args
    if not isinstance(x,np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) > 1 and x.shape[1] != 1:
        return None
    if (len(theta.shape) > 1 and theta.shape != (2, 1)) or (len(theta.shape) == 1 and theta.shape[0] != 2):
        return None
    if x.shape[-1] == 0 or theta.shape[-1] == 0:
        return None
    m = x.shape[0]
    y = np.zeros(m)
    try:
        for i, xi in enumerate(x):
            y[i] = theta[0] + theta[1] * xi
        return y
    except Exception:
        return None
        