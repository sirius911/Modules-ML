import functools
import numpy as np
import math

def noneValue(func):
    """ return None if Args is empty or not compatible with other function loss
    otherwise the result of function"""

    @functools.wraps(func)
    def function(*args, **kwargs):
        if not args or len(args) == 1 or len(args[1]) == 0:
            return None
        else:
            array1 = args[0]
            array2 = args[1]
            if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
                return None
            if len(array1.shape) > 1 and array1.shape[1] != 1:
                return None
            if array1.shape != array2.shape:
                return None
            return func(*args, **kwargs)
    return function

@noneValue
def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        loss_elem = (y_hat - y) * (y_hat - y)
        return loss_elem.sum() / len(y)
    except Exception:
        return None

@noneValue
def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        return math.sqrt(mse_(y, y_hat))
    except Exception:
        return None

@noneValue
def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        loss_elem = abs(y_hat - y)
        return loss_elem.sum() / len(y)
    except Exception:
        return None

@noneValue
def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        y_mean = y.mean()
        A = ((y_hat - y) * (y_hat - y))
        B = ((y - y_mean) * (y - y_mean))
        return (1 - (A.sum() / B.sum()))
    except Exception:
        return None