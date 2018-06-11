import numpy as np
from numba import jit
from .decorators import memorize

#if we want to go higher than this (unlikely) we have to change dtype from int to float (overflow)
TABLE_SIZE = 20

def fdouble_factorial(n):
    if n < 2:
        return 1
    else:
        return n * fdouble_factorial(n-2)


def ffactorial(n):
    if n < 2:
        return 1
    else:
        return n * ffactorial(n-1)

factorial = np.zeros(TABLE_SIZE, dtype=np.int64)
double_factorial = np.zeros(TABLE_SIZE, dtype=np.int64)
for i in range(TABLE_SIZE):
    factorial[i]        = ffactorial(i)
    double_factorial[i] = fdouble_factorial(i)




def detrace(tensor):
    dim = len(tensor.shape)
    tensor[np.diag_indices(dim)] -= tensor.trace() / dim
    return tensor


def dt2(x):
    r = np.einsum("ij...,...->ij...", np.eye(3), x.trace()/3)
    return x - r
