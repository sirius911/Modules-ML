import numpy as np
from loss import loss_, loss_elem_

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

y_hat=np.array([[1], [2], [3], [4]])
y=np.array([[0], [0], [0], [0]])
print(loss_(y,y))
print(loss_elem_(y,y))
print(loss_(y,y_hat))
print(loss_elem_(y, y_hat))

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
print(loss_(y1, y_hat1))
# Output:
# 3.0

x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
y_hat2 = predict_(x2, theta2)
y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
# Example 2:
print(loss_(y2, y_hat2))
# Output:
# 2.142857142857143
# Example 3:
print(loss_(y2, y2))
# Output:
# 0.0
