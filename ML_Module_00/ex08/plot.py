import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    prediction = predict_(x, theta)
    n = len(x)
    if prediction is not None:
        loss_elem = (prediction - y)*(prediction - y)
        # print(loss_elem)
        cost =loss_elem.sum() / len(y)
        plt.scatter(x, y, c='blue')
        plt.plot(x, prediction, color='red')
        for i in range(n):
            pA = [x[i],y[i]]
            pB = [x[i],prediction[i]]
            x__, y__ = [pA[0], pB[0]], [pA[1], pB[1]]
            plt.plot(x__, y__, 'r--')
        plt.title("Cost : "+str(cost))
        plt.show()