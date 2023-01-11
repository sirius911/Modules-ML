from ast import Div
from decimal import DivisionByZero
import random
from re import M
from matrix import Matrix, Vector, det
import numpy as np

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

def show_operation(m_x, m_y, res, ope):
    """show the operation with two Matrix"""
    print("")
    if isinstance(m_x, Matrix):
        m_x.show()
    else:
        print(f"\t{yellow}{m_x}{reset}")
    if len(ope) == 1:
        print(f"\t", end="")
    print(ope)
    if isinstance(m_y, Matrix):
        m_y.show()
    else:
        print(f"\t{yellow}{m_y}{reset}")
    print("______________")
    if isinstance(res, Matrix):
        res.show()
    else:
        print(f" {res}")
    print("")

def randmat(p, q, r=1):
    A = [q * [0] for i in range(p)]
    for i in range(p):
        for j in range(q):
            A[i][j] = random.uniform(-r, r)
    return A

print("\t\t███    ███  █████  ████████ ██████  ██ ██   ██ ")
print("\t\t████  ████ ██   ██    ██    ██   ██ ██  ██ ██  ")
print("\t\t██ ████ ██ ███████    ██    ██████  ██   ███   ")
print("\t\t██  ██  ██ ██   ██    ██    ██   ██ ██  ██ ██  ")
print("\t\t██      ██ ██   ██    ██    ██   ██ ██ ██   ██ ")
print("\nm1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])")
m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m1.show()
print(f"\tm1.shape = {green}{m1.shape}{reset}")
print(f"\n*** {yellow}m1.T(){reset} ***")
m1.T()
m1.show()
print(f"\tm1.shape = {green}{m1.shape}{reset}")
print(f"\n*** {yellow}m1.T(){reset} ***")
m1.T()
m1.show()
print(f"\tm1.shape = {green}{m1.shape}{reset}\n")

print("m2 = Matrix((3, 2))")
m2 = Matrix((3, 2))
m2.show()
print(f"\tShape -> {green}{m2.shape}{reset}")

print("m3 = Matrix((2, 3))")
m3 = Matrix((2, 3))
m3.show()
print(f"\tshape -> {green}{m3.shape}{reset}")

print("m4 = Matrix([[1,2,-1],[-2,1,1],[0,3,-3]])")
m4 = Matrix([[1,2,-1],[-2,1,1],[0,3,-3]])
m4.show()
print(f"\t-> is a square ? : {m4.square()}\t-> det = {det(m4.data)}")

print(f"\tshape -> {green}{m4.shape}{reset}")

input("Tap ENTER")
print("              _     _ _ _   _             ")
print("     /\      | |   | (_) | (_)            ")
print("    /  \   __| | __| |_| |_ _  ___  _ __  ")
print("   / /\ \ / _` |/ _` | | __| |/ _ \| '_ \ ")
print("  / ____ \ (_| | (_| | | |_| | (_) | | | |")
print(" /_/    \_\__,_|\__,_|_|\__|_|\___/|_| |_|")

m1_np = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
m2_np = np.array([[1,1,1],[2,2,2],[3,3,3]])
m1_ma = Matrix([[1, 2, 3], [4, 5, 6], [7,8,9]])
m2_ma = Matrix([[1,1,1],[2,2,2],[3,3,3]])
result_np = m1_np + m2_np
result_ma = m1_ma + m2_ma
print(f"\n\t******* {yellow}2  M A T R I X (same shape){reset}*******")
show_operation(m1_ma, m2_ma, result_ma, "+")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

result_np = m1_np + 5.0
result_ma = m1_ma + 5.0
print(f"\n\t******* {yellow}with a Scalar{reset} *******")
show_operation(m1_ma, 5, result_ma, "+")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

result_np = 5.0 + m1_np
result_ma = 5.0 + m1_ma
show_operation(5, m1_ma, result_ma, "+")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

m1_np = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
m2_np = np.array([[1,1,1],[2,2,2]])
m1_ma = Matrix([[1, 2, 3], [4, 5, 6], [7,8,9]])
m2_ma = Matrix([[1,1,1],[2,2,2]])
try:
    result_np = m1_np + m2_np
except ValueError:
    result_np = red + "Value Error" + reset
try:
    result_ma = m1_ma + m2_ma
except ValueError:
    result_ma = red + "Value Error" + reset
print(f"\n\t******* {yellow}2  M A T R I X (diff shape){reset}*******")
show_operation(m1_ma, m2_ma, result_ma, "+")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

input("Tap ENTER")
print("            _         _                  _   _             ")
print("           | |       | |                | | (_)            ")
print("  ___ _   _| |__  ___| |_ _ __ __ _  ___| |_ _  ___  _ __  ")
print(" / __| | | | '_ \/ __| __| '__/ _` |/ __| __| |/ _ \| '_ \ ")
print(" \__ \ |_| | |_) \__ \ |_| | | (_| | (__| |_| | (_) | | | |")
print(" |___/\__,_|_.__/|___/\__|_|  \__,_|\___|\__|_|\___/|_| |_|")

m1_np = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
m2_np = np.array([[1,1,1],[2,2,2],[3,3,3]])
m1_ma = Matrix([[1, 2, 3], [4, 5, 6], [7,8,9]])
m2_ma = Matrix([[1,1,1],[2,2,2],[3,3,3]])
result_np = m1_np - m2_np
result_ma = m1_ma - m2_ma
print(f"\n\t******* {yellow}2  M A T R I X {reset}*******")
show_operation(m1_ma, m2_ma, result_ma, "-")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"{m1_ma} - {m2_ma} = {result_ma} -> {okko(result_ma, result_np)}")
result1 = m1_ma.__rsub__(m2_ma)
result2 = m2_ma - m1_ma
show_operation(m1_ma, m2_ma, result_ma, "__rsub__")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"Matrix1.__rsub__(Maxtrix2) ->{okko(result1, result2)}")

print(f"\n\t******* {yellow}with a Scalar{reset} *******")
result_np = m1_np - 1.5
result_ma = m1_ma - 1.5
show_operation(m1_ma, 1.5, result_ma, "-")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"Matrix - 1.5 = {result_ma} -> {okko(result_ma, result_np)}")

result_np = 1.5- m1_np
result_ma = 1.5- m1_ma
show_operation(1.5, m1_ma, result_ma, "-")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"1.5 - Matrix = {result_ma} -> {okko(result_ma, result_np)}")

input("Tap ENTER")
print("                  _ _   _       _ _           _   _             ")
print("                 | | | (_)     | (_)         | | (_)            ")
print("  _ __ ___  _   _| | |_ _ _ __ | |_  ___ __ _| |_ _  ___  _ __  ")
print(" | '_ ` _ \| | | | | __| | '_ \| | |/ __/ _` | __| |/ _ \| '_ \ ")
print(" | | | | | | |_| | | |_| | |_) | | | (_| (_| | |_| | (_) | | | |")
print(" |_| |_| |_|\__,_|_|\__|_| .__/|_|_|\___\__,_|\__|_|\___/|_| |_|")
print("                         | |                                    ")
print("                         |_|                                    ")

print(f"\t******* {yellow}2  M A T R I X diff shape{reset}*******")
m1_ma = Matrix([[0.0, 1.0, 2.0, 3.0],[0.0, 2.0, 4.0, 6.0]])
m2_ma = Matrix([[0.0, 1.0],[2.0, 3.0],[4.0, 5.0],[6.0, 7.0]])
m1_np = np.array([[0.0, 1.0, 2.0, 3.0],[0.0, 2.0, 4.0, 6.0]])
m2_np = np.array([[0.0, 1.0],[2.0, 3.0],[4.0, 5.0],[6.0, 7.0]])
result_np = m1_np.dot(m2_np)
result_ma = m1_ma * m2_ma
show_operation(m1_ma, m2_ma, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"{m1_ma} x {m2_ma} = {result_ma} -> {okko(result_ma, result_np)}")

m1_ma = Matrix([[2,3,5,7],[0,9,3,7], [7,1,3,4]])
m2_ma = Matrix([[5,6],[4,7],[2,0],[0,1]])
m1_np = np.array([[2,3,5,7],[0,9,3,7], [7,1,3,4]])
m2_np = np.array([[5,6],[4,7],[2,0],[0,1]])
result_np = m1_np.dot(m2_np)    # produit matriciel
result_ma = m1_ma * m2_ma
show_operation(m1_ma, m2_ma, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

print(f"\t******* {yellow}2  M A T R I X (subject){reset}*******")
m1_ma = Matrix([[0.0, 1.0, 2.0, 3.0],[0.0, 2.0, 4.0, 6.0]])
m2_ma = Matrix([[0.0, 1.0],[2.0, 3.0],[4.0, 5.0],[6.0, 7.0]])
result_ma = m1_ma * m2_ma
show_operation(m1_ma, m2_ma, result_ma, "x")
print(f"{type(result_ma)}\tVS subject-> {okko(result_ma, Matrix([[28., 34.], [56., 68.]]))}")

print(f"\n\t******* {yellow}2  M A T R I X incompatible shape{reset}*******")
m2_ma = Matrix([[2,3,5,7],[0,9,3,7], [7,1,3,4]])
m1_ma = Matrix([[5,6],[4,7],[2,0],[0,1]])
m2_np = np.array([[2,3,5,7],[0,9,3,7], [7,1,3,4]])
m1_np = np.array([[5,6],[4,7],[2,0],[0,1]])
try:
    result_np = m1_np * m2_np
except ValueError as e:
    result_np = red + "ValueError" +reset
try:
    result_ma = m1_ma * m2_ma
except ValueError as e:
    result_ma = red + "ValueError" + reset
show_operation(m1_ma, m2_ma, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"{m1_ma} x {m2_ma} = {result_ma} -> {okko(result_ma, result_np)}")

print(f"\n\t******* {yellow}2  M A T R I X same shape (Term-to-term multiplication){reset}*******")
m1_ma = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2_ma = Matrix([[1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])

m1_np = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2_np = np.array([[1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
result_np = m1_np * m2_np
result_ma = m1_ma * m2_ma
show_operation(m1_ma, m2_ma, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"{m1_ma} x {m2_ma} = {result_ma} -> {okko(result_ma, result_np)}")

print(f"\n\t******* {yellow}with a Scalar{reset} *******")
m1_np = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
m1_ma = Matrix([[1, 2, 3], [4, 5, 6], [7,8,9]])
result_np = m1_np * 1.5
result_ma = m1_ma * 1.5
show_operation(m1_ma, 1.5, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"Matrix * 1.5 -> {okko(result_ma,result_np)}")

result_np = 1.5 * m1_np
result_ma = 1.5 * m1_ma
show_operation(1.5, m1_ma, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")
# print(f"1.5 * Matrix-> {okko(result_ma,result_np)}")

input("Tap ENTER")
print("      _ _       _     _             ")
print("     | (_)     (_)   (_)            ")
print("   __| |___   ___ ___ _  ___  _ __  ")
print("  / _` | \ \ / / / __| |/ _ \| '_ \ ")
print(" | (_| | |\ V /| \__ \ | (_) | | | |")
print("  \__,_|_| \_/ |_|___/_|\___/|_| |_|")

print(f"\t******* {yellow}2  M A T R I X diff shape{reset}*******")

m1_ma = Matrix(randmat(3,3))
m2_ma = Matrix(randmat(3,2))
m1_np = np.array(m1_ma.data)
m2_np = np.array(m2_ma.data)

try:
    result_np = m1_np / m2_np
except ValueError as e:
    print(e)
    result_np = red + "ValueError" +reset
try:
    result_ma = m1_ma / m2_ma
except ValueError as e:
    result_ma = red + "ValueError" + reset
show_operation(m1_ma, m2_ma, result_ma, "x")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

print(f"\n\t******* {yellow}2  M A T R I X same shape (Term-to-term multiplication){reset}*******")
m1_ma = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2_ma = Matrix([[1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
m1_np = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2_np = np.array([[1.5, 2.0], [2.5, 3.0], [3.5, 4.0]])
result_np = m1_np / m2_np
result_ma = m1_ma / m2_ma
show_operation(m1_ma, m2_ma, result_ma, "/")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

print(f"\n\t******* {yellow}2  M A T R I X same shape (Div by zero){reset}*******")
m1_ma = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
m2_ma = Matrix([[1.5, 0.0], [2.5, 3.0], [3.5, 4.0]])

try:
    result_ma = m1_ma / m2_ma
except DivisionByZero:
    result_ma = red + "Division by zero" + reset

show_operation(m1_ma, m2_ma, result_ma, "/")
print(f"\t -> {okko(result_ma, red + 'Division by zero' + reset)}")

print(f"\n\t******* {yellow}with a Scalar{reset} *******")
m1_np = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
m1_ma = Matrix([[1, 2, 3], [4, 5, 6], [7,8,9]])
result_np = m1_np / 1.5
result_ma = m1_ma / 1.5
show_operation(m1_ma, 1.5, result_ma, "/")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

result_np = 1.5 / m1_np
result_ma = 1.5 / m1_ma
show_operation(1.5, m1_ma, result_ma, "/")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

result_np = 0.0 / m1_np
result_ma = 0.0 / m1_ma
show_operation(0.0, m1_ma, result_ma, "/")
print(f"\t VS numpy-> {okko(result_ma, result_np)}")

try:
    result_ma = m1_ma / 0.0
except DivisionByZero as e:
    result_ma = red + "Division by zero" + reset
show_operation(m1_ma, 0.0, result_ma, "/")
print(f"\t VS numpy-> {okko(result_ma, red + 'Division by zero' + reset)}")

input("Tape ENTER")
print("\n██    ██ ███████  ██████ ████████  ██████  ██████  ")
print("██    ██ ██      ██         ██    ██    ██ ██   ██ ")
print("██    ██ █████   ██         ██    ██    ██ ██████  ")
print(" ██  ██  ██      ██         ██    ██    ██ ██   ██ ")
print("  ████   ███████  ██████    ██     ██████  ██   ██ ")
print("\nv1 = Vector([[1, 2, 3]]) ")
v1 = Vector([[1, 2, 3]])
print(f"\t{v1} ->{green}{v1.shape}{reset}")
print(f"\tv1.T() = {v1.T()} -> {green}{v1.shape}{reset}")

print("v2 = Vector([[1], [2], [3]])")
v2 = Vector([[1], [2], [3]])
print(f"\t{v2} ->{green}{v2.shape}{reset}")
print(f"\tv2.T() = {v2.T()} -> {green}{v2.shape}{reset}")

print("v3 = Vector([[1, 2], [3, 4]]) # return an error")
try:
    v3 = Vector([[1, 2], [3, 4]]) # return an error
except Exception as e:
    print(e)

print("              _     _ _ _   _             ")
print("     /\      | |   | (_) | (_)            ")
print("    /  \   __| | __| |_| |_ _  ___  _ __  ")
print("   / /\ \ / _` |/ _` | | __| |/ _ \| '_ \ ")
print("  / ____ \ (_| | (_| | | |_| | (_) | | | |")
print(" /_/    \_\__,_|\__,_|_|\__|_|\___/|_| |_|")

v1_mv = Vector([[1], [2], [3]])
v2_mv = Vector([[2], [4], [8]])
v1_np = np.array([[1], [2], [3]])
v2_np = np.array([[2], [4], [8]])
result_np = v1_np + v2_np
result_mv = v1_mv + v2_mv
print(v1_mv)
print("\t+")
print(v2_mv)
print("_______________________")
print(f"{result_mv} VS numpy -> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[4, 5, 6]])
v1_np = np.array([[1, 2, 3]])
v2_np = np.array([[4, 5, 6]])
result_np = v1_np + v2_np
result_mv = v1_mv + v2_mv
print(v1_mv)
print("\t+")
print(v2_mv)
print("_______________________")
print(f"{result_mv} VS numpy -> {okko(result_mv, result_np)}\n")

print(f"\n\t******* {yellow}with a Scalar{reset} *******")
v1_mv = Vector([[1], [2], [3]])
v1_np = np.array([[1], [2], [3]])
result_np = v1_np + 5.0
result_mv = v1_mv + 5.0
print(f"\nVector + 5.0 = {result_mv} VS numpy-> {okko(result_mv, result_np)}")

result_np = 5.0 + v1_np
result_mv = 5.0 + v1_mv
print(f"5.0 + Vector = {result_mv} VS numpy -> {okko(result_mv, result_np)}\n")

print(f"\n\t******* {yellow}with diff shape{reset} *******")
v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[2], [4], [8]])
#Numpy make a matrix forbidden in subject
result_mv = v1_mv + v2_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t+")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[3], [6], [11]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")

v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[2], [4], [8]])
result_mv = v2_mv + v1_mv
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("\t+")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[3], [6], [11]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")

v1_mv = Vector([[1, 2, 3]]) 
v1_np = np.array([[1, 2, 3]])
result_np = v1_np + v1_np
result_mv = v1_mv + v1_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t+")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[2], [4], [8]]) 
v1_np = np.array([[2], [4], [8]])
result_np = v1_np + v1_np
result_mv = v1_mv + v1_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t+")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[2], [4], [8], [10]])
try:
    result_mv = v1_mv + v2_mv
except ValueError as e:
    result_mv = red + str(e) + reset
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t+")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv}\n")

input("Tap ENTER")
print("            _         _                  _   _             ")
print("           | |       | |                | | (_)            ")
print("  ___ _   _| |__  ___| |_ _ __ __ _  ___| |_ _  ___  _ __  ")
print(" / __| | | | '_ \/ __| __| '__/ _` |/ __| __| |/ _ \| '_ \ ")
print(" \__ \ |_| | |_) \__ \ |_| | | (_| | (__| |_| | (_) | | | |")
print(" |___/\__,_|_.__/|___/\__|_|  \__,_|\___|\__|_|\___/|_| |_|")

v1_mv = Vector([[1], [2], [3]])
v2_mv = Vector([[2], [4], [8]])
v1_np = np.array([[1], [2], [3]])
v2_np = np.array([[2], [4], [8]])
result_np = v1_np - v2_np
result_mv = v1_mv - v2_mv
print(v1_mv)
print("\t-")
print(v2_mv)
print("_______________________")
print(f"{result_mv} VS numpy -> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[4, 5, 6]])
v1_np = np.array([[1, 2, 3]])
v2_np = np.array([[4, 5, 6]])
result_np = v1_np - v2_np
result_mv = v1_mv - v2_mv
print(v1_mv)
print("\t-")
print(v2_mv)
print("_______________________")
print(f"{result_mv} VS numpy -> {okko(result_mv, result_np)}\n")

print(f"\n\t******* {yellow}with a Scalar{reset} *******")
result_np = v1_np - 1.0
result_mv = v1_mv - 1.0
print(f"Vector - 1.0 = {result_mv} -> {okko(result_mv, result_np)}")

result_np = 5.0 - v1_np
result_mv = 5.0 - v1_mv 
print(f"5.0 - Vector = {result_mv} -> {okko(result_mv, result_np)}")

print(f"\n\t******* {yellow}with diff shape{reset} *******")
v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[2], [4], [8]])

result_mv = v1_mv - v2_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t-")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[-1], [-2], [-5]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")

v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[2], [4], [8]])
result_mv = v2_mv - v1_mv
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("\t-")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[1], [2], [5]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")


v1_mv = Vector([[1, 2, 3]]) 
v1_np = np.array([[1, 2, 3]])
result_np = v1_np - v1_np
result_mv = v1_mv - v1_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t-")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[2], [4], [8]]) 
v1_np = np.array([[2], [4], [8]])
result_np = v1_np - v1_np
result_mv = v1_mv - v1_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t-")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1, 2, 3]]) 
v2_mv = Vector([[2], [4], [8], [10]])
try:
    result_mv = v1_mv - v2_mv
except ValueError as e:
    result_mv = red + str(e) + reset
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t-")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv}\n")

input("Tap ENTER")

print("                  _ _   _       _ _           _   _             ")
print("                 | | | (_)     | (_)         | | (_)            ")
print("  _ __ ___  _   _| | |_ _ _ __ | |_  ___ __ _| |_ _  ___  _ __  ")
print(" | '_ ` _ \| | | | | __| | '_ \| | |/ __/ _` | __| |/ _ \| '_ \ ")
print(" | | | | | | |_| | | |_| | |_) | | | (_| (_| | |_| | (_) | | | |")
print(" |_| |_| |_|\__,_|_|\__|_| .__/|_|_|\___\__,_|\__|_|\___/|_| |_|")
print("                         | |                                    ")
print("                         |_|                                    ")
v1_mv = Vector([[1], [2], [3]])
v2_mv = Vector([[2], [4], [8]])
v1_np = np.array([[1], [2], [3]])
v2_np = np.array([[2], [4], [8]])
result_np = v1_np * v2_np
result_mv = v1_mv * v2_mv
print(f"\n\t******* {yellow}with same shape{reset} *******")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\tX")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1],[2],[3]])
v2_mv = Vector([[2, 4, 8]])
result_mv = v1_mv * v2_mv
print(f"\n\t******* {yellow}with diff shape{reset} *******")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\tX")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[2], [8], [24]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")


v1_mv = Vector([[1],[2],[3]])
v2_mv = Vector([[2, 4, 8]])
result_mv =  v2_mv * v1_mv
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("\tX")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv,Vector([[2], [8], [24]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")

v1_mv = Vector([[1],[2],[3],[5]])
v2_mv = Vector([[2, 4, 8]])
try:
    result_mv =  v2_mv * v1_mv
except ValueError as e:
    result_mv = red + str(e) + reset

print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("\tX")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv}\n")

print(f"\n\t******* {yellow}with a Scalar{reset} *******")
v1_mv = Vector([[1],[2],[3],[5]])
v1_np = np.array([[1],[2],[3],[5]])
result_np = v1_np * 2.5
result_mv = v1_mv * 2.5
print(f"\t Vector * 2.5 = {result_mv} -> {okko(result_mv, result_np)}")

result_np = 2.5 * v1_np
result_mv = 2.5 * v1_mv
print(f"\t 2.5 * Vector = {result_mv} -> {okko(result_mv, result_np)}")

print("      _ _       _     _             ")
print("     | (_)     (_)   (_)            ")
print("   __| |___   ___ ___ _  ___  _ __  ")
print("  / _` | \ \ / / / __| |/ _ \| '_ \ ")
print(" | (_| | |\ V /| \__ \ | (_) | | | |")
print("  \__,_|_| \_/ |_|___/_|\___/|_| |_|")
print(f"\n\t******* {yellow}with same shape{reset} *******")
v1_mv = Vector([[1], [2], [3]])
v2_mv = Vector([[2], [4], [8]])
v1_np = np.array([[1], [2], [3]])
v2_np = np.array([[2], [4], [8]])
result_np = v1_np / v2_np
result_mv = v1_mv / v2_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t /")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1, 2, 3]])
v2_mv = Vector([[2, 4, 8]])
v1_np = np.array([[1, 2, 3]])
v2_np = np.array([[2, 4, 8]])
result_np = v1_np / v2_np
result_mv = v1_mv / v2_mv
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t /")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy --> {okko(result_mv, result_np)}\n")

v1_mv = Vector([[1],[2],[3]])
v2_mv = Vector([[2, 4, 1]])
result_mv = v1_mv / v2_mv
print(f"\n\t******* {yellow}with diff shape{reset} *******")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t /")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[0.5], [0.5], [3]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")

v1_mv = Vector([[2, 4, 1]])
v2_mv = Vector([[1],[2],[4]])
result_mv = v1_mv / v2_mv
print(f"\n\t******* {yellow}with diff shape{reset} *******")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t /")
print(f"{v2_mv} {yellow}{v2_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} --> {okko(result_mv, Vector([[2.0], [2.0], [0.25]]))}\t{yellow}'Numpy make a matrix forbidden in subject'{reset}\n")


v1_mv = Vector([[2, 4, 1]])
v1_np = np.array([[2, 4, 1]])
result_np = v1_np / 2.0
result_mv = v1_mv / 2.0
print(f"\n\t******* {yellow}with a Scalar{reset} *******")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("\t /")
print(f"\t 2.0")
print("_______________________")
print(f"{result_mv} VS numpy--> {okko(result_mv, result_np)}\n")

result_np = 2.0 / v1_np
result_mv = 2.0 / v1_mv
print(f"\n\t 2.0")
print("\t /")
print(f"{v1_mv} {yellow}{v1_mv.shape}{reset}")
print("_______________________")
print(f"{result_mv} VS numpy--> {okko(result_mv, result_np)}\n")




print("███    ███  █████  ██   ██ ████████ ██████  ██ ██   ██     ██   ██     ██    ██ ███████  ██████ ████████  ██████  ██████  ")
print("████  ████ ██   ██  ██ ██     ██    ██   ██ ██  ██ ██       ██ ██      ██    ██ ██      ██         ██    ██    ██ ██   ██ ")
print("██ ████ ██ ███████   ███      ██    ██████  ██   ███         ███       ██    ██ █████   ██         ██    ██    ██ ██████  ")
print("██  ██  ██ ██   ██  ██ ██     ██    ██   ██ ██  ██ ██       ██ ██       ██  ██  ██      ██         ██    ██    ██ ██   ██ ")
print("██      ██ ██   ██ ██   ██    ██    ██   ██ ██ ██   ██     ██   ██       ████   ███████  ██████    ██     ██████  ██   ██ ")
m1_ma = Matrix([[0.0, 1.0, 2.0],[0.0, 2.0, 4.0]])
v1_mv = Vector([[1], [2], [3]])
m1_np = np.array([[0.0, 1.0, 2.0],[0.0, 2.0, 4.0]])
v1_np = np.array([[1], [2], [3]])
result_ma = m1_ma * v1_mv
result_np = m1_np.dot(v1_np)
print(f"\n\t******* {yellow}mutiplication subject{reset} *******")
m1_ma.show()
print("\t X")
v1_mv.show()
print("_______________________")
print(f"{result_ma} VS numpy.dot--> {okko(result_ma, result_np)}\n")

m1_ma = Matrix([[0.0, 1.0, 2.0],[0.0, 2.0, 4.0]])
v1_mv = Vector([[1, 2, 3]])
m1_np = np.array([[0.0, 1.0, 2.0],[0.0, 2.0, 4.0]])
v1_np = np.array([1, 2, 3])
result_ma =  m1_ma * v1_mv
result_np = m1_np.dot(v1_np)
print(f"\n\t******* {yellow}mutiplication subject{reset} *******")
m1_ma.show()
print("\t X")
v1_mv.show()
print("_______________________")
print(f"{result_ma} VS subject--> {okko(result_ma, Vector([[8.0],[16.0]]))}\n")
