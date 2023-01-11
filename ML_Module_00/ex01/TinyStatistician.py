import functools
import math
import numpy as np

def noneValue(func):
    """ return None if Args is empty
    otherwise the result of function"""

    @functools.wraps(func)
    def function(*args, **kwargs):
        if not args or len(args) == 1 or len(args[1]) == 0:
            return None
        else:
            if isinstance(args[1], list):
                for el in args[1]:
                    if not isinstance(el, (int,float)):
                        return None
            elif isinstance(args[1], np.ndarray):
                tab = args[1]
                if len(tab.shape) == 1:
                    for el in tab:
                        if not isinstance(el, (np.int64,np.float64)):
                            return None
                else:
                    nb_line, nb_col = tab.shape
                    for i in range(nb_line):
                        for j in range(nb_col):
                            if not isinstance(tab[i][j], (np.int64, np.float64)):
                                return None
            else:
                return None
            return func(*args, **kwargs)
    return function

class TinyStatistician:
    """ Initiation to vey basic statistic operation"""

    def __init__(self):
        """ initialisation"""
        pass

    @noneValue
    def mean(self, x):
        """
            computes the mean of a given non-empty list or array x, using a for-loop.
            args:
                non-empty list or array
            return;
                float
        """
        sum = 0
        if type(x) == list:
            for el in x:
                sum += el
            return float(sum / len(x))
        else:
            if len(x.shape) == 1:
                for el in x:
                    sum += el
                return float(sum / len(x))    
            for line in x:
                for col in line:
                    sum += col
            return float(sum / x.size)

    @noneValue
    def median(self, array):
        """ computes the median of a given non-empty list or array
            return the median as a float
            or None if the list or array is empty
        """
        array_sorted = sorted(array)
        n = len(array_sorted)
        if n % 2 == 0:
            rang1 = int(n/2)
            rang2 = int((n / 2) + 1)
            valeur1 = array_sorted[rang1 - 1]
            valeur2 = array_sorted[rang2 - 1]
            valeur = (valeur1 + valeur2) / 2
        else:
            rang = int((n + 1) / 2)
            valeur = array_sorted[rang - 1]
        return float(valeur)
    
    @noneValue
    def quartile(self, array):
        """computes the 1st and 3th quartiles of a given
        non-empty list or array
        return tuple of float or None if list or array empty
        """
        result = [0.0, 0.0]
        array_sorted = sorted(array)
        n = len(array_sorted)
        rang_q1 = ((n + 3) / 4)
        if int(rang_q1) == rang_q1:
            result[0] = float(array_sorted[int(rang_q1 - 1)])
        else:
            coef = rang_q1 - int(rang_q1)
            r_inf = array_sorted[int(rang_q1 - 1)]
            r_sup = array_sorted[int(rang_q1)]
            # print(f"coef = {coef}\tr_inf = {r_inf}\tr_sup = {r_sup}")
            if coef == 0.25:
                # result[0] = ((r_inf * 3) + r_sup) / 4
                result[0] = r_inf
            elif coef == 0.75:
                result[0] = r_sup
            else:
                result[0] = (r_inf + r_sup) / 2
        
        rang_q3 = ((3 * n) + 1) / 4
        if int(rang_q3) == rang_q3:
            result[1] = float(array_sorted[int(rang_q3 - 1)])
        else:
            coef = rang_q3 - int(rang_q3)
            r_inf = array_sorted[int(rang_q3 - 1)]
            r_sup = array_sorted[int(rang_q3)]
            # print(f"coef = {coef}\tr_inf = {r_inf}\tr_sup = {r_sup}")
            
            if coef == 0.25:
                result[1] = r_inf
            elif coef == 0.75:
                result[1] = r_sup
            else:
                result[1] = (r_inf + r_sup) / 4
        return result

    @noneValue
    def var(self, array):
        """computes the variance of a given non-empty list or array"""
        mu = self.mean(array)
        m = len(array)
        sum = 0
        for x in array:
            sum += ((x - mu) * (x - mu))
        return round(sum * 1/(m - 1))

    @noneValue
    def std(self, array):
        """ computes the standard deviation of a given non-empty list pr array"""
        return round(math.sqrt(self.var(array)), 2)

    @noneValue
    def  percentile(self, array, p):
        """
            computes the expected percentile of a given non-empty list or
            array x. The method returns the percentile as a float, otherwise None if x is an
            empty list or array or a non expected type object. The second parameter is the
            wished percentile. This method should not raise any Exception.

        """
        array_sorted = sorted(array)
        n = len(array_sorted)
        rang_ordinal = math.ceil(p * n / 100)
        return (array_sorted[rang_ordinal - 1])