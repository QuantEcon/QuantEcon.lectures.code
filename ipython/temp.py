## Filename: temp.py
import numpy as np

def p1(x, coef):
    return sum(a * x**i for i, a in enumerate(coef))

def p2(x, coef):
    X = np.empty(len(coef))
    X[0] = 1
    X[1:] = x
    y = np.cumprod(X)   # y = [1, x, x**2,...]
    return np.dot(coef, y)