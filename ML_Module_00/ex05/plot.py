import numpy as np
import matplotlib.pyplot as plt

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    # control args
    if not isinstance(x,np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) > 1 and x.shape[1] != 1:
        return None
    if (len(theta.shape) > 1 and theta.shape != (2, 1)) or (len(theta.shape) == 1 and theta.shape[0] != 2):
        return None
    if x.shape[-1] == 0 or theta.shape[-1] == 0:
        return None

    x_1 = np.c_[np.ones(x.shape[0]), x]
    return x_1.dot(theta)

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """
    prediction = predict_(x, theta)
    if prediction is not None:
        plt.scatter(x, y, c='blue')
        plt.plot(x, prediction, color='red')
        plt.show()