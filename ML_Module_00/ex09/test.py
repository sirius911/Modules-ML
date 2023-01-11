import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from other_losses import mae_, mse_, r2score_, rmse_

green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'
yellow = '\033[33m'

def okko(resultA, resultB):
    """
    return Ok or KO if resultA equal or not equal with resultB
    """
    if resultA == resultB:
        return (green + "OK" + reset)
    else:
        return (red + "KO" + reset)

x_ = np.array([0, 15, -9, 7, 12, 3, -21])
y_ = np.array([2, 14, -13, 5, 12, 4, -19])

print("********** M S E *************")
y_hat=np.array([[1], [2], [3], [4]])
y=np.array([[0], [0], [0], [0]])
print(mse_(y,y))
print(mse_(y,y_hat))

slk = mean_squared_error(x_,y_)
ft =mse_(x_,y_)
print(f"mse = {ft} -> {okko(ft, slk)}")


print("************** RM S E **************")
print(rmse_(y,y))
print(rmse_(y,y_hat))
slk = sqrt(mean_squared_error(x_,y_))
ft = rmse_(x_,y_)
print(f"rmse = {ft} -> {okko(ft, slk)}")


print("************* M A E **********")
print(mae_(y,y))
print(mae_(y,y_hat))
slk = mean_absolute_error(x_,y_)
ft = mae_(x_, y_)
print(f"mae = {ft} -> {okko(ft, slk)}")


print("********** R2   S C O R E   ***********")
print(r2_score(y,y))
print(r2_score(y_hat,y))
slk = r2_score(x_,y_)
ft = r2score_(x_,y_)
print(f"r2score = {ft} -> {okko(ft, slk)}")
print(r2_score(y_hat,y))
