import array
from copy import deepcopy
from decimal import DivisionByZero
from distutils.log import error
import numbers


  
def det(m):
    def matrice_cut(m,i):
     
        mbis = deepcopy(m)
        del(mbis[i])
        for line in mbis:
            del(line[0])
        return mbis
 
    return m[0][0] if len(m)==1 else sum(((-1)**i)*m[i][0]*det(matrice_cut(m,i)) \
           for i in range(0,len(m)))

class Matrix:
    """
    Matrix Class
    """

    def __init__(self, arg):
        if isinstance(arg, list):
            self.data = arg
            rows = len(self.data)
            cols = len(self.data[0])
        elif isinstance(arg, tuple):
            rows = arg[0]
            cols = arg[1]
            self.data = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    row.append(0.0)
                self.data.append(row)
        else:
            raise TypeError("Error __init__() of Matrix")
        self.shape = (rows, cols)
        self.id = id(self)

    def T(self):
        """
         Returns the transpose vector
         (i.e. a column vector into a row vector,
         or a row vector into a column vector
        """
        new_value = []
        for c in range(self.shape[1]):
            row = []
            for r in range(self.shape[0]):
                row.append(self.data[r][c])
            new_value.append(row)
        self.data = new_value
        self.shape = (self.shape[1], self.shape[0])
        return self

    
    def __add__(self, x):
        """ Addition between two matrices of same dimension (m x n)"""
        if isinstance(x, Matrix):
            if x.shape == self.shape:
                m = self.shape[1]
                n = self.shape[0]
                new = [[self.data[i][j] + x.data[i][j] for j in range(m)] for i in range(n)]
                return Matrix(new)        
            m = self.shape[0]
            n = self.shape[1]
            p = x.shape[1]
            if n != x.shape[0]:
                raise ValueError("operands could not be broadcast together with shapes "+str(self.shape) + " "+str(x.shape))
            #(m × n) and (n × p)
            new = [[0]*p for i in range(m)] # matrix (m * p)
            # print(new)
            for i in range(m):
                for j in range(p):
                    for k in range(n):
                        new[i][j] = self.data[i][k] + x.data[k][j]
            return Matrix(new)
        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Addition by something other than a scalar is not defined here !")
        m = self.shape[1]
        n = self.shape[0]
        new = [[self.data[i][j] + scalar for j in range(m)] for i in range(n)]
        return Matrix(new)

    def __radd__(self, matrix):
        """Revers Addition"""
        return self.__add__(matrix)

    def __sub__(self, x):
        """substraction of two matrix with same shape"""
        if isinstance(x, Matrix):
            if x.shape == self.shape:
                m = self.shape[1]
                n = self.shape[0]
                new = [[self.data[i][j] - x.data[i][j] for j in range(m)] for i in range(n)]
                return Matrix(new)
            m = self.shape[0]
            n = self.shape[1]
            p = x.shape[1]
            if n != x.shape[0]:
                raise ValueError("operands could not be broadcast together with shapes "+str(self.shape) + " "+str(x.shape))
            #(m × n) and (n × p)
            new = [[0]*p for i in range(m)] # matrix (m * p)
            # print(new)
            for i in range(m):
                for j in range(p):
                    for k in range(n):
                        new[i][j] = self.data[i][k] - x.data[k][j]
            return Matrix(new)
        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Substraction by something other than a scalar is not defined here !")
        m = self.shape[1]
        n = self.shape[0]
        new = [[self.data[i][j] - scalar for j in range(m)] for i in range(n)]
        return Matrix(new)


    def __rsub__(self, x):
        """revers substraction"""
        if isinstance(x, Matrix):
            if x.shape == self.shape:
                m = self.shape[1]
                n = self.shape[0]
                new = [[x.data[i][j] - self.data[i][j]  for j in range(m)] for i in range(n)]
                return Matrix(new)
            m = self.shape[0]
            n = self.shape[1]
            p = x.shape[1]
            if n != x.shape[0]:
                raise ValueError("operands could not be broadcast together with shapes "+str(self.shape) + " "+str(x.shape))
            #(m × n) and (n × p)
            new = [[0]*p for i in range(m)] # matrix (m * p)
            # print(new)
            for i in range(m):
                for j in range(p):
                    for k in range(n):
                        new[i][j] = x.data[k][j] - self.data[i][k]
            return Matrix(new)
        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Substraction by something other than a scalar is not defined here !")
        m = self.shape[1]
        n = self.shape[0]
        new = [[scalar - self.data[i][j] for j in range(m)] for i in range(n)]
        return Matrix(new)

    def __truediv__(self, x):
        """ division"""
        print("lalalalal")
        if isinstance(x, Matrix):
            if self.shape == x.shape:
                # => Term-to-term division
                m = self.shape[1]
                n = self.shape[0]
                try:
                    new = [[self.data[i][j] / x.data[i][j] for j in range(m)] for i in range(n)]
                except Exception:
                    raise DivisionByZero("RuntimeWarning: divide by zero encountered in true_divide")
                return Matrix(new)
            else:
                raise ValueError("operands could not be broadcast together with shapes "+str(self.shape) + " "+str(x.shape))

        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Multiplication by something other than a scalar is not defined here !")
        if scalar == 0:
            raise DivisionByZero("Division by zero")
        m = self.shape[1]
        n = self.shape[0]
        new = [[self.data[i][j] / scalar for j in range(m)] for i in range(n)]
        return Matrix(new)

    def __rtruediv__(self, x):
        """invers division"""
        if isinstance(x, Matrix):
            if x.shape == self.shape:
                m = self.shape[1]
                n = self.shape[0]
                new = [[x.data[i][j] / self.data[i][j]  for j in range(m)] for i in range(n)]
                return Matrix(new)
            m = self.shape[0]
            n = self.shape[1]
            p = x.shape[1]
            if n != x.shape[0]:
                raise ValueError("operands could not be broadcast together with shapes "+str(self.shape) + " "+str(x.shape))
            #(m × n) and (n × p)
            new = [[0]*p for i in range(m)] # matrix (m * p)
            print(new)
            for i in range(m):
                for j in range(p):
                    for k in range(n):
                        new[i][j] = x.data[k][j] / self.data[i][k]
            return Matrix(new)
        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Multiplication by something other than a scalar is not defined here !")
        m = self.shape[1]
        n = self.shape[0]
        new = [[scalar / self.data[i][j] for j in range(m)] for i in range(n)]
        return Matrix(new)

    def __mul__(self, x):
        """ multiplication """

        if isinstance(x, Matrix):
            if self.shape == x.shape:
                # => Term-to-term multiplication
                m = self.shape[1]
                n = self.shape[0]
                new = [[self.data[i][j] * x.data[i][j] for j in range(m)] for i in range(n)]
                return Matrix(new)
            m = self.shape[0]
            n = self.shape[1]
            p = x.shape[1]
            if isinstance(x, Vector) and n == p:
                # (m * n) * (1 * n) --> (m * n ) * (n * 1)
                p = 1
                cpy = Vector(x.data)
                cpy.T()
                new = [[0]*p for i in range(m)] # matrix (m * p)
                # print(new)
                for i in range(m):
                    for j in range(p):
                        for k in range(n):
                            new[i][j] += self.data[i][k] * cpy.data[k][j]
                return Vector(new)
            if n != x.shape[0]:
                raise ValueError("operands could not be broadcast together with shapes "+str(self.shape) + " "+str(x.shape))
                        #(m × n) and (n × p)
            new = [[0]*p for i in range(m)] # matrix (m * p)
            # print(new)
            for i in range(m):
                for j in range(p):
                    for k in range(n):
                        new[i][j] += self.data[i][k] * x.data[k][j]
            if isinstance(x, Vector):
                return Vector(new)
            return Matrix(new)
        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Multiplication by something other than a scalar is not defined here !")
        m = self.shape[1]
        n = self.shape[0]
        new = [[self.data[i][j] * scalar for j in range(m)] for i in range(n)]
        return Matrix(new)

    def __rmul__(self, x):
        """reverse multiplication"""
        return self.__mul__(x)

    def __eq__(self, __o):
        if "shape" not in dir(__o):
            return False
        if __o.shape == self.shape:
            for j in range(self.shape[1]):
                for i in range(self.shape[0]):
                    a = self.data[i][j]
                    b = __o[i][j]
                    if a != b:
                        return False
        else:
            return False
        return True

    def __ne__(self, __o):
        return not self.__eq__(__o)

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return (f"Matrix({str(self.data)})")

    def __repr__(self):
        return (self.__str__())

    def show(self):
        print("[", end="")
        for i in range(self.shape[0]):
            if i>0:
                print(" ", end="")
            print(f"{self.data[i]}, ", end="")
            if i < self.shape[0]:
                print("")
            else:
                print("]")

    def square(self):
        """ return True if the Matrix is a square (m * m) False otherwise"""
        return self.shape[0] == self.shape[1]

    def eye(self):
        """ return the identity matrix if it is a square Matrix or None otherwise"""
        if self.square():
            n = self.shape[0]
            A = [n * [0] for i in range(n)]
            for i in range(n):
                A[i][i] = 1
            return A
        else:
            return None
    
    def is_inversible(self):
        """ return True if the Matrix is inversible"""
        return self.square and det(self.data) != 0

class Vector(Matrix):
    """Class Vector"""

    def __init__(self, arg):
        super().__init__(arg)
        if not (self.shape[0] == 1 or self.shape[1] == 1):
            raise ValueError("Error: the shape "+str(self.shape) + " is not correct for Vector.")

    def dot(self, v):
        """return a dot product between two vectors of same shape"""
        if not isinstance(v, Vector) or v.shape != self.shape:
            raise ValueError("ValueError: The dot product must be with a vector of same shape !")
        dot = 0
        if self.shape[0] == 1:
            #dot with shape 1 * n
            for a,b in zip(self.data[0], v.data[0]):
                dot += a * b
        else:
            #dot with shape n * 1
            for a,b in zip(self.data, v.data):
                dot += a[0] * b[0]
        return dot

    def __mul__(self, x):
        """ Multiplication between two vectors of same dimension (m x n)"""
        if isinstance(x, Vector):
            if self.shape == x.shape:
                return Vector(super().__mul__(x).data)
            else:
                if self.shape[0] == 1 and x.shape[1] == 1 and \
                    self.shape[1] == x.shape[0]:
                    cpy = Vector(self.data)
                    return x.__mul__(cpy.T())
                elif self.shape[1] == 1 and x.shape[0] == 1 and \
                    self.shape[0] == x.shape[1]:
                    cpy = Vector(x.data)
                    return cpy.T().__mul__(self)
                else:
                    raise ValueError("Error: shapes not compatible.")
            # m = self.shape[1]
            # n = self.shape[0]
            # if x.shape[0] == m and x.shape[1] == n: # (m x n) and (n x m)
            #     if m == 1:  # (1 * m) * (m * 1)
            #         return Matrix(super().__mul__(x).data)
            #     return x.__mul__(self)
            # new = [[self.data[i][j] * x.data[i][j] for j in range(m)] for i in range(n)]
            # return Vector(new)
        try:
            scalar = float(x)
        except:
            txt = "NotImplementedError: Multiplication by " +  str(type(x)) + "is not defined here !"
            raise NotImplementedError(txt)
        m = self.shape[1]
        n = self.shape[0]
        new = [[self.data[i][j] * scalar for j in range(m)] for i in range(n)]
        return Vector(new)

    def __truediv__(self, x):
        """division"""
        if isinstance(x, Vector):
            if self.shape == x.shape:
                return Vector(super().__truediv__(x).data)
            else:
                if self.shape[0] == 1 and x.shape[1] == 1 and \
                    self.shape[1] == x.shape[0]:
                    cpy = Vector(self.data)
                    return x.__rtruediv__(cpy.T())
                elif self.shape[1] == 1 and x.shape[0] == 1 and \
                    self.shape[0] == x.shape[1]:
                    cpy = Vector(x.data)
                    return cpy.T().__rtruediv__(self)
                else:
                    raise ValueError("Error: shapes not compatible.")
        try:
            scalar = float(x)
        except:
            raise NotImplementedError("NotImplementedError: Multiplication by something other than a scalar is not defined here !")
        print("icicicici")
        m = self.shape[1]
        n = self.shape[0]
        new = [[self.data[i][j] / scalar for j in range(m)] for i in range(n)]
        return Vector(new)

    def __add__(self, x):
        if isinstance(x, Vector):
            if self.shape == x.shape:
                return Vector(super().__add__(x).data)
            else:
                if self.shape[0] == 1 and x.shape[1] == 1 and \
                    self.shape[1] == x.shape[0]:
                    cpy = Vector(self.data)
                    return x.__add__(cpy.T())
                elif self.shape[1] == 1 and x.shape[0] == 1 and \
                    self.shape[0] == x.shape[1]:
                    cpy = Vector(x.data)
                    return cpy.T().__add__(self)
                else:
                    raise ValueError("Error: shapes not compatible.")
            
        else:
            return Vector(super().__add__(x).data)

    def __sub__(self, x):
        if isinstance(x, Vector):
            if self.shape == x.shape:
                return Vector(super().__sub__(x).data)
            else:
                if self.shape[0] == 1 and x.shape[1] == 1 and \
                    self.shape[1] == x.shape[0]:
                    cpy = Vector(self.data)
                    return Vector(x.__rsub__(cpy.T()).data)
                elif self.shape[1] == 1 and x.shape[0] == 1 and \
                    self.shape[0] == x.shape[1]:
                    cpy = Vector(x.data)
                    return Vector(cpy.T().__rsub__(self).data)
                else:
                    raise ValueError("Error: shapes not compatible.")
        else:
            return Vector(super().__sub__(x).data)

    def __str__(self):
        return (f"Vector({str(self.data)})")

    def __repr__(self):
        return (f"Vector({str(self.data)})")
        