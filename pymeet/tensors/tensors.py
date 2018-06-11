from __future__ import division
from numba import jit
from numpy import sqrt
import numpy as np
@jit
def T0(x,y,z):
    return np.array([1/sqrt(x**2 + y**2 + z**2)])
@jit
def Tx(x,y,z):
    return -x/(x**2 + y**2 + z**2)**(3/2)
@jit
def Ty(x,y,z):
    return -y/(x**2 + y**2 + z**2)**(3/2)
@jit
def Tz(x,y,z):
    return -z/(x**2 + y**2 + z**2)**(3/2)
@jit
def T1(x,y,z):
    arr = np.zeros((3,), dtype=np.float64)
    arr[(0,)] = Tx(x,y,z)
    arr[(1,)] = Ty(x,y,z)
    arr[(2,)] = Tz(x,y,z)
    return arr
@jit
def Txx(x,y,z):
    return (3*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(3/2)
@jit
def Txy(x,y,z):
    return 3*x*y/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txz(x,y,z):
    return 3*x*z/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyx(x,y,z):
    return 3*x*y/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyy(x,y,z):
    return (3*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(3/2)
@jit
def Tyz(x,y,z):
    return 3*y*z/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzx(x,y,z):
    return 3*x*z/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzy(x,y,z):
    return 3*y*z/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzz(x,y,z):
    return (3*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(3/2)
@jit
def T2(x,y,z):
    arr = np.zeros((3, 3), dtype=np.float64)
    arr[(0, 0)] = Txx(x,y,z)
    arr[(0, 1)] = Txy(x,y,z)
    arr[(0, 2)] = Txz(x,y,z)
    arr[(1, 0)] = Tyx(x,y,z)
    arr[(1, 1)] = Tyy(x,y,z)
    arr[(1, 2)] = Tyz(x,y,z)
    arr[(2, 0)] = Tzx(x,y,z)
    arr[(2, 1)] = Tzy(x,y,z)
    arr[(2, 2)] = Tzz(x,y,z)
    return arr
@jit
def Txxx(x,y,z):
    return 3*x*(-5*x**2/(x**2 + y**2 + z**2) + 3)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txxy(x,y,z):
    return 3*y*(-5*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txxz(x,y,z):
    return 3*z*(-5*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txyx(x,y,z):
    return 3*y*(-5*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txyy(x,y,z):
    return 3*x*(-5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txyz(x,y,z):
    return -15*x*y*z/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzx(x,y,z):
    return 3*z*(-5*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txzy(x,y,z):
    return -15*x*y*z/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzz(x,y,z):
    return 3*x*(-5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyxx(x,y,z):
    return 3*y*(-5*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyxy(x,y,z):
    return 3*x*(-5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyxz(x,y,z):
    return -15*x*y*z/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyx(x,y,z):
    return 3*x*(-5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyyy(x,y,z):
    return 3*y*(-5*y**2/(x**2 + y**2 + z**2) + 3)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyyz(x,y,z):
    return 3*z*(-5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyzx(x,y,z):
    return -15*x*y*z/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzy(x,y,z):
    return 3*z*(-5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyzz(x,y,z):
    return 3*y*(-5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzxx(x,y,z):
    return 3*z*(-5*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzxy(x,y,z):
    return -15*x*y*z/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxz(x,y,z):
    return 3*x*(-5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzyx(x,y,z):
    return -15*x*y*z/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyy(x,y,z):
    return 3*z*(-5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzyz(x,y,z):
    return 3*y*(-5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzzx(x,y,z):
    return 3*x*(-5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzzy(x,y,z):
    return 3*y*(-5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzzz(x,y,z):
    return 3*z*(-5*z**2/(x**2 + y**2 + z**2) + 3)/(x**2 + y**2 + z**2)**(5/2)
@jit
def T3(x,y,z):
    arr = np.zeros((3, 3, 3), dtype=np.float64)
    arr[(0, 0, 0)] = Txxx(x,y,z)
    arr[(0, 0, 1)] = Txxy(x,y,z)
    arr[(0, 0, 2)] = Txxz(x,y,z)
    arr[(0, 1, 0)] = Txyx(x,y,z)
    arr[(0, 1, 1)] = Txyy(x,y,z)
    arr[(0, 1, 2)] = Txyz(x,y,z)
    arr[(0, 2, 0)] = Txzx(x,y,z)
    arr[(0, 2, 1)] = Txzy(x,y,z)
    arr[(0, 2, 2)] = Txzz(x,y,z)
    arr[(1, 0, 0)] = Tyxx(x,y,z)
    arr[(1, 0, 1)] = Tyxy(x,y,z)
    arr[(1, 0, 2)] = Tyxz(x,y,z)
    arr[(1, 1, 0)] = Tyyx(x,y,z)
    arr[(1, 1, 1)] = Tyyy(x,y,z)
    arr[(1, 1, 2)] = Tyyz(x,y,z)
    arr[(1, 2, 0)] = Tyzx(x,y,z)
    arr[(1, 2, 1)] = Tyzy(x,y,z)
    arr[(1, 2, 2)] = Tyzz(x,y,z)
    arr[(2, 0, 0)] = Tzxx(x,y,z)
    arr[(2, 0, 1)] = Tzxy(x,y,z)
    arr[(2, 0, 2)] = Tzxz(x,y,z)
    arr[(2, 1, 0)] = Tzyx(x,y,z)
    arr[(2, 1, 1)] = Tzyy(x,y,z)
    arr[(2, 1, 2)] = Tzyz(x,y,z)
    arr[(2, 2, 0)] = Tzzx(x,y,z)
    arr[(2, 2, 1)] = Tzzy(x,y,z)
    arr[(2, 2, 2)] = Tzzz(x,y,z)
    return arr
@jit
def Txxxx(x,y,z):
    return 3*(35*x**4/(x**2 + y**2 + z**2)**2 - 30*x**2/(x**2 + y**2 + z**2) + 3)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txxxy(x,y,z):
    return 15*x*y*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxz(x,y,z):
    return 15*x*z*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyx(x,y,z):
    return 15*x*y*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyy(x,y,z):
    return 3*(35*x**2*y**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txxyz(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzx(x,y,z):
    return 15*x*z*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzy(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzz(x,y,z):
    return 3*(35*x**2*z**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txyxx(x,y,z):
    return 15*x*y*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxy(x,y,z):
    return 3*(35*x**2*y**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txyxz(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyx(x,y,z):
    return 3*(35*x**2*y**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txyyy(x,y,z):
    return 15*x*y*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyz(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzx(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzy(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzz(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxx(x,y,z):
    return 15*x*z*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxy(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxz(x,y,z):
    return 3*(35*x**2*z**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txzyx(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyy(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyz(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzx(x,y,z):
    return 3*(35*x**2*z**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Txzzy(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzz(x,y,z):
    return 15*x*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxx(x,y,z):
    return 15*x*y*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxy(x,y,z):
    return 3*(35*x**2*y**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyxxz(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyx(x,y,z):
    return 3*(35*x**2*y**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyxyy(x,y,z):
    return 15*x*y*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyz(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzx(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzy(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzz(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxx(x,y,z):
    return 3*(35*x**2*y**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyyxy(x,y,z):
    return 15*x*y*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxz(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyx(x,y,z):
    return 15*x*y*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyy(x,y,z):
    return 3*(35*y**4/(x**2 + y**2 + z**2)**2 - 30*y**2/(x**2 + y**2 + z**2) + 3)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyyyz(x,y,z):
    return 15*y*z*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzx(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzy(x,y,z):
    return 15*y*z*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzz(x,y,z):
    return 3*(35*y**2*z**2/(x**2 + y**2 + z**2)**2 - 5*y**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyzxx(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxy(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxz(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyx(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyy(x,y,z):
    return 15*y*z*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyz(x,y,z):
    return 3*(35*y**2*z**2/(x**2 + y**2 + z**2)**2 - 5*y**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyzzx(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzy(x,y,z):
    return 3*(35*y**2*z**2/(x**2 + y**2 + z**2)**2 - 5*y**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tyzzz(x,y,z):
    return 15*y*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxx(x,y,z):
    return 15*x*z*(7*x**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxy(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxz(x,y,z):
    return 3*(35*x**2*z**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzxyx(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyy(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyz(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzx(x,y,z):
    return 3*(35*x**2*z**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzxzy(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzz(x,y,z):
    return 15*x*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxx(x,y,z):
    return 15*y*z*(7*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxy(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxz(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyx(x,y,z):
    return 15*x*z*(7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyy(x,y,z):
    return 15*y*z*(7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyz(x,y,z):
    return 3*(35*y**2*z**2/(x**2 + y**2 + z**2)**2 - 5*y**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzyzx(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzy(x,y,z):
    return 3*(35*y**2*z**2/(x**2 + y**2 + z**2)**2 - 5*y**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzyzz(x,y,z):
    return 15*y*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxx(x,y,z):
    return 3*(35*x**2*z**2/(x**2 + y**2 + z**2)**2 - 5*x**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzzxy(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxz(x,y,z):
    return 15*x*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyx(x,y,z):
    return 15*x*y*(7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyy(x,y,z):
    return 3*(35*y**2*z**2/(x**2 + y**2 + z**2)**2 - 5*y**2/(x**2 + y**2 + z**2) - 5*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(5/2)
@jit
def Tzzyz(x,y,z):
    return 15*y*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzx(x,y,z):
    return 15*x*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzy(x,y,z):
    return 15*y*z*(7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzz(x,y,z):
    return 3*(35*z**4/(x**2 + y**2 + z**2)**2 - 30*z**2/(x**2 + y**2 + z**2) + 3)/(x**2 + y**2 + z**2)**(5/2)
@jit
def T4(x,y,z):
    arr = np.zeros((3, 3, 3, 3), dtype=np.float64)
    arr[(0, 0, 0, 0)] = Txxxx(x,y,z)
    arr[(0, 0, 0, 1)] = Txxxy(x,y,z)
    arr[(0, 0, 0, 2)] = Txxxz(x,y,z)
    arr[(0, 0, 1, 0)] = Txxyx(x,y,z)
    arr[(0, 0, 1, 1)] = Txxyy(x,y,z)
    arr[(0, 0, 1, 2)] = Txxyz(x,y,z)
    arr[(0, 0, 2, 0)] = Txxzx(x,y,z)
    arr[(0, 0, 2, 1)] = Txxzy(x,y,z)
    arr[(0, 0, 2, 2)] = Txxzz(x,y,z)
    arr[(0, 1, 0, 0)] = Txyxx(x,y,z)
    arr[(0, 1, 0, 1)] = Txyxy(x,y,z)
    arr[(0, 1, 0, 2)] = Txyxz(x,y,z)
    arr[(0, 1, 1, 0)] = Txyyx(x,y,z)
    arr[(0, 1, 1, 1)] = Txyyy(x,y,z)
    arr[(0, 1, 1, 2)] = Txyyz(x,y,z)
    arr[(0, 1, 2, 0)] = Txyzx(x,y,z)
    arr[(0, 1, 2, 1)] = Txyzy(x,y,z)
    arr[(0, 1, 2, 2)] = Txyzz(x,y,z)
    arr[(0, 2, 0, 0)] = Txzxx(x,y,z)
    arr[(0, 2, 0, 1)] = Txzxy(x,y,z)
    arr[(0, 2, 0, 2)] = Txzxz(x,y,z)
    arr[(0, 2, 1, 0)] = Txzyx(x,y,z)
    arr[(0, 2, 1, 1)] = Txzyy(x,y,z)
    arr[(0, 2, 1, 2)] = Txzyz(x,y,z)
    arr[(0, 2, 2, 0)] = Txzzx(x,y,z)
    arr[(0, 2, 2, 1)] = Txzzy(x,y,z)
    arr[(0, 2, 2, 2)] = Txzzz(x,y,z)
    arr[(1, 0, 0, 0)] = Tyxxx(x,y,z)
    arr[(1, 0, 0, 1)] = Tyxxy(x,y,z)
    arr[(1, 0, 0, 2)] = Tyxxz(x,y,z)
    arr[(1, 0, 1, 0)] = Tyxyx(x,y,z)
    arr[(1, 0, 1, 1)] = Tyxyy(x,y,z)
    arr[(1, 0, 1, 2)] = Tyxyz(x,y,z)
    arr[(1, 0, 2, 0)] = Tyxzx(x,y,z)
    arr[(1, 0, 2, 1)] = Tyxzy(x,y,z)
    arr[(1, 0, 2, 2)] = Tyxzz(x,y,z)
    arr[(1, 1, 0, 0)] = Tyyxx(x,y,z)
    arr[(1, 1, 0, 1)] = Tyyxy(x,y,z)
    arr[(1, 1, 0, 2)] = Tyyxz(x,y,z)
    arr[(1, 1, 1, 0)] = Tyyyx(x,y,z)
    arr[(1, 1, 1, 1)] = Tyyyy(x,y,z)
    arr[(1, 1, 1, 2)] = Tyyyz(x,y,z)
    arr[(1, 1, 2, 0)] = Tyyzx(x,y,z)
    arr[(1, 1, 2, 1)] = Tyyzy(x,y,z)
    arr[(1, 1, 2, 2)] = Tyyzz(x,y,z)
    arr[(1, 2, 0, 0)] = Tyzxx(x,y,z)
    arr[(1, 2, 0, 1)] = Tyzxy(x,y,z)
    arr[(1, 2, 0, 2)] = Tyzxz(x,y,z)
    arr[(1, 2, 1, 0)] = Tyzyx(x,y,z)
    arr[(1, 2, 1, 1)] = Tyzyy(x,y,z)
    arr[(1, 2, 1, 2)] = Tyzyz(x,y,z)
    arr[(1, 2, 2, 0)] = Tyzzx(x,y,z)
    arr[(1, 2, 2, 1)] = Tyzzy(x,y,z)
    arr[(1, 2, 2, 2)] = Tyzzz(x,y,z)
    arr[(2, 0, 0, 0)] = Tzxxx(x,y,z)
    arr[(2, 0, 0, 1)] = Tzxxy(x,y,z)
    arr[(2, 0, 0, 2)] = Tzxxz(x,y,z)
    arr[(2, 0, 1, 0)] = Tzxyx(x,y,z)
    arr[(2, 0, 1, 1)] = Tzxyy(x,y,z)
    arr[(2, 0, 1, 2)] = Tzxyz(x,y,z)
    arr[(2, 0, 2, 0)] = Tzxzx(x,y,z)
    arr[(2, 0, 2, 1)] = Tzxzy(x,y,z)
    arr[(2, 0, 2, 2)] = Tzxzz(x,y,z)
    arr[(2, 1, 0, 0)] = Tzyxx(x,y,z)
    arr[(2, 1, 0, 1)] = Tzyxy(x,y,z)
    arr[(2, 1, 0, 2)] = Tzyxz(x,y,z)
    arr[(2, 1, 1, 0)] = Tzyyx(x,y,z)
    arr[(2, 1, 1, 1)] = Tzyyy(x,y,z)
    arr[(2, 1, 1, 2)] = Tzyyz(x,y,z)
    arr[(2, 1, 2, 0)] = Tzyzx(x,y,z)
    arr[(2, 1, 2, 1)] = Tzyzy(x,y,z)
    arr[(2, 1, 2, 2)] = Tzyzz(x,y,z)
    arr[(2, 2, 0, 0)] = Tzzxx(x,y,z)
    arr[(2, 2, 0, 1)] = Tzzxy(x,y,z)
    arr[(2, 2, 0, 2)] = Tzzxz(x,y,z)
    arr[(2, 2, 1, 0)] = Tzzyx(x,y,z)
    arr[(2, 2, 1, 1)] = Tzzyy(x,y,z)
    arr[(2, 2, 1, 2)] = Tzzyz(x,y,z)
    arr[(2, 2, 2, 0)] = Tzzzx(x,y,z)
    arr[(2, 2, 2, 1)] = Tzzzy(x,y,z)
    arr[(2, 2, 2, 2)] = Tzzzz(x,y,z)
    return arr
@jit
def Txxxxx(x,y,z):
    return 15*x*(-63*x**4/(x**2 + y**2 + z**2)**2 + 70*x**2/(x**2 + y**2 + z**2) - 15)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxxy(x,y,z):
    return 45*y*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxxz(x,y,z):
    return 45*z*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxyx(x,y,z):
    return 45*y*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxyy(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxyz(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txxxzx(x,y,z):
    return 45*z*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxxzy(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txxxzz(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyxx(x,y,z):
    return 45*y*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyxy(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyxz(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txxyyx(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyyy(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyyz(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyzx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txxyzy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxyzz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzxx(x,y,z):
    return 45*z*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzxy(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txxzxz(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzyx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txxzyy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzyz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzzx(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzzy(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txxzzz(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxxx(x,y,z):
    return 45*y*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxxy(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxxz(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txyxyx(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxyy(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxyz(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxzx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txyxzy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyxzz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyxx(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyxy(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyxz(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyyx(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyyy(x,y,z):
    return 45*x*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyyz(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txyyzx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyyzy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txyyzz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzxx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txyzxy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzxz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzyx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzyy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txyzyz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzzx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzzy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txyzzz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzxxx(x,y,z):
    return 45*z*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxxy(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzxxz(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxyx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzxyy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxyz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxzx(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxzy(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzxzz(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyxx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzyxy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyxz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyyx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyyy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzyyz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyzx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyzy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzyzz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzzxx(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzxy(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzxz(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzyx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzyy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzyz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzzzx(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Txzzzy(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Txzzzz(x,y,z):
    return 45*x*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxxx(x,y,z):
    return 45*y*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxxy(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxxz(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyxxyx(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxyy(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxyz(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxzx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyxxzy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxxzz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyxx(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyxy(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyxz(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyyx(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyyy(x,y,z):
    return 45*x*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyyz(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyxyzx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxyzy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyxyzz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzxx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyxzxy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzxz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzyx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzyy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyxzyz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzzx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzzy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyxzzz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyxxx(x,y,z):
    return 15*x*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxxy(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxxz(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxyx(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxyy(x,y,z):
    return 45*x*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxyz(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyxzx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyxzy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyxzz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyxx(x,y,z):
    return 15*y*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyxy(x,y,z):
    return 45*x*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyxz(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyyyx(x,y,z):
    return 45*x*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyyy(x,y,z):
    return 15*y*(-63*y**4/(x**2 + y**2 + z**2)**2 + 70*y**2/(x**2 + y**2 + z**2) - 15)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyyz(x,y,z):
    return 45*z*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyzx(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyyzy(x,y,z):
    return 45*z*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyyzz(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzxx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzxy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyzxz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzyx(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyyzyy(x,y,z):
    return 45*z*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzyz(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzzx(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzzy(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyyzzz(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxxx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzxxy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxxz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxyx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxyy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzxyz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxzx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxzy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzxzz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzyxx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyxy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzyxz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyyx(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzyyy(x,y,z):
    return 45*z*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyyz(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyzx(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyzy(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzyzz(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzxx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzxy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzxz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzzyx(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzyy(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzyz(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzzx(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tyzzzy(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tyzzzz(x,y,z):
    return 45*y*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxxx(x,y,z):
    return 45*z*(-21*x**4/(x**2 + y**2 + z**2)**2 + 14*x**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxxy(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxxxz(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxyx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxxyy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxyz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxzx(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxzy(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxxzz(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyxx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxyxy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyxz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyyx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyyy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxyyz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyzx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyzy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxyzz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxzxx(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzxy(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzxz(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzyx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzyy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzyz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxzzx(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzxzzy(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzxzzz(x,y,z):
    return 45*x*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxxx(x,y,z):
    return 315*x*y*z*(-3*x**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyxxy(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxxz(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxyx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxyy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyxyz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxzx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxzy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyxzz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyyxx(x,y,z):
    return 15*z*(-63*x**2*y**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyxy(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyyxz(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyyx(x,y,z):
    return 315*x*y*z*(-3*y**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyyyy(x,y,z):
    return 45*z*(-21*y**4/(x**2 + y**2 + z**2)**2 + 14*y**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyyz(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyzx(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyzy(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyyzz(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzxx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzxy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzxz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyzyx(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzyy(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzyz(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzzx(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzyzzy(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzyzzz(x,y,z):
    return 45*y*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxxx(x,y,z):
    return 15*x*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxxy(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxxz(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxyx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxyy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxyz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzzxzx(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzxzy(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzzxzz(x,y,z):
    return 45*x*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyxx(x,y,z):
    return 15*y*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 7*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyxy(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyxz(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzzyyx(x,y,z):
    return 15*x*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyyy(x,y,z):
    return 15*y*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 7*y**2/(x**2 + y**2 + z**2) + 21*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyyz(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyzx(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzzyzy(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzyzz(x,y,z):
    return 45*y*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzxx(x,y,z):
    return 15*z*(-63*x**2*z**2/(x**2 + y**2 + z**2)**2 + 21*x**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzxy(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzzzxz(x,y,z):
    return 45*x*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzyx(x,y,z):
    return 315*x*y*z*(-3*z**2/(x**2 + y**2 + z**2) + 1)/(x**2 + y**2 + z**2)**(9/2)
@jit
def Tzzzyy(x,y,z):
    return 15*z*(-63*y**2*z**2/(x**2 + y**2 + z**2)**2 + 21*y**2/(x**2 + y**2 + z**2) + 7*z**2/(x**2 + y**2 + z**2) - 3)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzyz(x,y,z):
    return 45*y*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzzx(x,y,z):
    return 45*x*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzzy(x,y,z):
    return 45*y*(-21*z**4/(x**2 + y**2 + z**2)**2 + 14*z**2/(x**2 + y**2 + z**2) - 1)/(x**2 + y**2 + z**2)**(7/2)
@jit
def Tzzzzz(x,y,z):
    return 15*z*(-63*z**4/(x**2 + y**2 + z**2)**2 + 70*z**2/(x**2 + y**2 + z**2) - 15)/(x**2 + y**2 + z**2)**(7/2)
@jit
def T5(x,y,z):
    arr = np.zeros((3, 3, 3, 3, 3), dtype=np.float64)
    arr[(0, 0, 0, 0, 0)] = Txxxxx(x,y,z)
    arr[(0, 0, 0, 0, 1)] = Txxxxy(x,y,z)
    arr[(0, 0, 0, 0, 2)] = Txxxxz(x,y,z)
    arr[(0, 0, 0, 1, 0)] = Txxxyx(x,y,z)
    arr[(0, 0, 0, 1, 1)] = Txxxyy(x,y,z)
    arr[(0, 0, 0, 1, 2)] = Txxxyz(x,y,z)
    arr[(0, 0, 0, 2, 0)] = Txxxzx(x,y,z)
    arr[(0, 0, 0, 2, 1)] = Txxxzy(x,y,z)
    arr[(0, 0, 0, 2, 2)] = Txxxzz(x,y,z)
    arr[(0, 0, 1, 0, 0)] = Txxyxx(x,y,z)
    arr[(0, 0, 1, 0, 1)] = Txxyxy(x,y,z)
    arr[(0, 0, 1, 0, 2)] = Txxyxz(x,y,z)
    arr[(0, 0, 1, 1, 0)] = Txxyyx(x,y,z)
    arr[(0, 0, 1, 1, 1)] = Txxyyy(x,y,z)
    arr[(0, 0, 1, 1, 2)] = Txxyyz(x,y,z)
    arr[(0, 0, 1, 2, 0)] = Txxyzx(x,y,z)
    arr[(0, 0, 1, 2, 1)] = Txxyzy(x,y,z)
    arr[(0, 0, 1, 2, 2)] = Txxyzz(x,y,z)
    arr[(0, 0, 2, 0, 0)] = Txxzxx(x,y,z)
    arr[(0, 0, 2, 0, 1)] = Txxzxy(x,y,z)
    arr[(0, 0, 2, 0, 2)] = Txxzxz(x,y,z)
    arr[(0, 0, 2, 1, 0)] = Txxzyx(x,y,z)
    arr[(0, 0, 2, 1, 1)] = Txxzyy(x,y,z)
    arr[(0, 0, 2, 1, 2)] = Txxzyz(x,y,z)
    arr[(0, 0, 2, 2, 0)] = Txxzzx(x,y,z)
    arr[(0, 0, 2, 2, 1)] = Txxzzy(x,y,z)
    arr[(0, 0, 2, 2, 2)] = Txxzzz(x,y,z)
    arr[(0, 1, 0, 0, 0)] = Txyxxx(x,y,z)
    arr[(0, 1, 0, 0, 1)] = Txyxxy(x,y,z)
    arr[(0, 1, 0, 0, 2)] = Txyxxz(x,y,z)
    arr[(0, 1, 0, 1, 0)] = Txyxyx(x,y,z)
    arr[(0, 1, 0, 1, 1)] = Txyxyy(x,y,z)
    arr[(0, 1, 0, 1, 2)] = Txyxyz(x,y,z)
    arr[(0, 1, 0, 2, 0)] = Txyxzx(x,y,z)
    arr[(0, 1, 0, 2, 1)] = Txyxzy(x,y,z)
    arr[(0, 1, 0, 2, 2)] = Txyxzz(x,y,z)
    arr[(0, 1, 1, 0, 0)] = Txyyxx(x,y,z)
    arr[(0, 1, 1, 0, 1)] = Txyyxy(x,y,z)
    arr[(0, 1, 1, 0, 2)] = Txyyxz(x,y,z)
    arr[(0, 1, 1, 1, 0)] = Txyyyx(x,y,z)
    arr[(0, 1, 1, 1, 1)] = Txyyyy(x,y,z)
    arr[(0, 1, 1, 1, 2)] = Txyyyz(x,y,z)
    arr[(0, 1, 1, 2, 0)] = Txyyzx(x,y,z)
    arr[(0, 1, 1, 2, 1)] = Txyyzy(x,y,z)
    arr[(0, 1, 1, 2, 2)] = Txyyzz(x,y,z)
    arr[(0, 1, 2, 0, 0)] = Txyzxx(x,y,z)
    arr[(0, 1, 2, 0, 1)] = Txyzxy(x,y,z)
    arr[(0, 1, 2, 0, 2)] = Txyzxz(x,y,z)
    arr[(0, 1, 2, 1, 0)] = Txyzyx(x,y,z)
    arr[(0, 1, 2, 1, 1)] = Txyzyy(x,y,z)
    arr[(0, 1, 2, 1, 2)] = Txyzyz(x,y,z)
    arr[(0, 1, 2, 2, 0)] = Txyzzx(x,y,z)
    arr[(0, 1, 2, 2, 1)] = Txyzzy(x,y,z)
    arr[(0, 1, 2, 2, 2)] = Txyzzz(x,y,z)
    arr[(0, 2, 0, 0, 0)] = Txzxxx(x,y,z)
    arr[(0, 2, 0, 0, 1)] = Txzxxy(x,y,z)
    arr[(0, 2, 0, 0, 2)] = Txzxxz(x,y,z)
    arr[(0, 2, 0, 1, 0)] = Txzxyx(x,y,z)
    arr[(0, 2, 0, 1, 1)] = Txzxyy(x,y,z)
    arr[(0, 2, 0, 1, 2)] = Txzxyz(x,y,z)
    arr[(0, 2, 0, 2, 0)] = Txzxzx(x,y,z)
    arr[(0, 2, 0, 2, 1)] = Txzxzy(x,y,z)
    arr[(0, 2, 0, 2, 2)] = Txzxzz(x,y,z)
    arr[(0, 2, 1, 0, 0)] = Txzyxx(x,y,z)
    arr[(0, 2, 1, 0, 1)] = Txzyxy(x,y,z)
    arr[(0, 2, 1, 0, 2)] = Txzyxz(x,y,z)
    arr[(0, 2, 1, 1, 0)] = Txzyyx(x,y,z)
    arr[(0, 2, 1, 1, 1)] = Txzyyy(x,y,z)
    arr[(0, 2, 1, 1, 2)] = Txzyyz(x,y,z)
    arr[(0, 2, 1, 2, 0)] = Txzyzx(x,y,z)
    arr[(0, 2, 1, 2, 1)] = Txzyzy(x,y,z)
    arr[(0, 2, 1, 2, 2)] = Txzyzz(x,y,z)
    arr[(0, 2, 2, 0, 0)] = Txzzxx(x,y,z)
    arr[(0, 2, 2, 0, 1)] = Txzzxy(x,y,z)
    arr[(0, 2, 2, 0, 2)] = Txzzxz(x,y,z)
    arr[(0, 2, 2, 1, 0)] = Txzzyx(x,y,z)
    arr[(0, 2, 2, 1, 1)] = Txzzyy(x,y,z)
    arr[(0, 2, 2, 1, 2)] = Txzzyz(x,y,z)
    arr[(0, 2, 2, 2, 0)] = Txzzzx(x,y,z)
    arr[(0, 2, 2, 2, 1)] = Txzzzy(x,y,z)
    arr[(0, 2, 2, 2, 2)] = Txzzzz(x,y,z)
    arr[(1, 0, 0, 0, 0)] = Tyxxxx(x,y,z)
    arr[(1, 0, 0, 0, 1)] = Tyxxxy(x,y,z)
    arr[(1, 0, 0, 0, 2)] = Tyxxxz(x,y,z)
    arr[(1, 0, 0, 1, 0)] = Tyxxyx(x,y,z)
    arr[(1, 0, 0, 1, 1)] = Tyxxyy(x,y,z)
    arr[(1, 0, 0, 1, 2)] = Tyxxyz(x,y,z)
    arr[(1, 0, 0, 2, 0)] = Tyxxzx(x,y,z)
    arr[(1, 0, 0, 2, 1)] = Tyxxzy(x,y,z)
    arr[(1, 0, 0, 2, 2)] = Tyxxzz(x,y,z)
    arr[(1, 0, 1, 0, 0)] = Tyxyxx(x,y,z)
    arr[(1, 0, 1, 0, 1)] = Tyxyxy(x,y,z)
    arr[(1, 0, 1, 0, 2)] = Tyxyxz(x,y,z)
    arr[(1, 0, 1, 1, 0)] = Tyxyyx(x,y,z)
    arr[(1, 0, 1, 1, 1)] = Tyxyyy(x,y,z)
    arr[(1, 0, 1, 1, 2)] = Tyxyyz(x,y,z)
    arr[(1, 0, 1, 2, 0)] = Tyxyzx(x,y,z)
    arr[(1, 0, 1, 2, 1)] = Tyxyzy(x,y,z)
    arr[(1, 0, 1, 2, 2)] = Tyxyzz(x,y,z)
    arr[(1, 0, 2, 0, 0)] = Tyxzxx(x,y,z)
    arr[(1, 0, 2, 0, 1)] = Tyxzxy(x,y,z)
    arr[(1, 0, 2, 0, 2)] = Tyxzxz(x,y,z)
    arr[(1, 0, 2, 1, 0)] = Tyxzyx(x,y,z)
    arr[(1, 0, 2, 1, 1)] = Tyxzyy(x,y,z)
    arr[(1, 0, 2, 1, 2)] = Tyxzyz(x,y,z)
    arr[(1, 0, 2, 2, 0)] = Tyxzzx(x,y,z)
    arr[(1, 0, 2, 2, 1)] = Tyxzzy(x,y,z)
    arr[(1, 0, 2, 2, 2)] = Tyxzzz(x,y,z)
    arr[(1, 1, 0, 0, 0)] = Tyyxxx(x,y,z)
    arr[(1, 1, 0, 0, 1)] = Tyyxxy(x,y,z)
    arr[(1, 1, 0, 0, 2)] = Tyyxxz(x,y,z)
    arr[(1, 1, 0, 1, 0)] = Tyyxyx(x,y,z)
    arr[(1, 1, 0, 1, 1)] = Tyyxyy(x,y,z)
    arr[(1, 1, 0, 1, 2)] = Tyyxyz(x,y,z)
    arr[(1, 1, 0, 2, 0)] = Tyyxzx(x,y,z)
    arr[(1, 1, 0, 2, 1)] = Tyyxzy(x,y,z)
    arr[(1, 1, 0, 2, 2)] = Tyyxzz(x,y,z)
    arr[(1, 1, 1, 0, 0)] = Tyyyxx(x,y,z)
    arr[(1, 1, 1, 0, 1)] = Tyyyxy(x,y,z)
    arr[(1, 1, 1, 0, 2)] = Tyyyxz(x,y,z)
    arr[(1, 1, 1, 1, 0)] = Tyyyyx(x,y,z)
    arr[(1, 1, 1, 1, 1)] = Tyyyyy(x,y,z)
    arr[(1, 1, 1, 1, 2)] = Tyyyyz(x,y,z)
    arr[(1, 1, 1, 2, 0)] = Tyyyzx(x,y,z)
    arr[(1, 1, 1, 2, 1)] = Tyyyzy(x,y,z)
    arr[(1, 1, 1, 2, 2)] = Tyyyzz(x,y,z)
    arr[(1, 1, 2, 0, 0)] = Tyyzxx(x,y,z)
    arr[(1, 1, 2, 0, 1)] = Tyyzxy(x,y,z)
    arr[(1, 1, 2, 0, 2)] = Tyyzxz(x,y,z)
    arr[(1, 1, 2, 1, 0)] = Tyyzyx(x,y,z)
    arr[(1, 1, 2, 1, 1)] = Tyyzyy(x,y,z)
    arr[(1, 1, 2, 1, 2)] = Tyyzyz(x,y,z)
    arr[(1, 1, 2, 2, 0)] = Tyyzzx(x,y,z)
    arr[(1, 1, 2, 2, 1)] = Tyyzzy(x,y,z)
    arr[(1, 1, 2, 2, 2)] = Tyyzzz(x,y,z)
    arr[(1, 2, 0, 0, 0)] = Tyzxxx(x,y,z)
    arr[(1, 2, 0, 0, 1)] = Tyzxxy(x,y,z)
    arr[(1, 2, 0, 0, 2)] = Tyzxxz(x,y,z)
    arr[(1, 2, 0, 1, 0)] = Tyzxyx(x,y,z)
    arr[(1, 2, 0, 1, 1)] = Tyzxyy(x,y,z)
    arr[(1, 2, 0, 1, 2)] = Tyzxyz(x,y,z)
    arr[(1, 2, 0, 2, 0)] = Tyzxzx(x,y,z)
    arr[(1, 2, 0, 2, 1)] = Tyzxzy(x,y,z)
    arr[(1, 2, 0, 2, 2)] = Tyzxzz(x,y,z)
    arr[(1, 2, 1, 0, 0)] = Tyzyxx(x,y,z)
    arr[(1, 2, 1, 0, 1)] = Tyzyxy(x,y,z)
    arr[(1, 2, 1, 0, 2)] = Tyzyxz(x,y,z)
    arr[(1, 2, 1, 1, 0)] = Tyzyyx(x,y,z)
    arr[(1, 2, 1, 1, 1)] = Tyzyyy(x,y,z)
    arr[(1, 2, 1, 1, 2)] = Tyzyyz(x,y,z)
    arr[(1, 2, 1, 2, 0)] = Tyzyzx(x,y,z)
    arr[(1, 2, 1, 2, 1)] = Tyzyzy(x,y,z)
    arr[(1, 2, 1, 2, 2)] = Tyzyzz(x,y,z)
    arr[(1, 2, 2, 0, 0)] = Tyzzxx(x,y,z)
    arr[(1, 2, 2, 0, 1)] = Tyzzxy(x,y,z)
    arr[(1, 2, 2, 0, 2)] = Tyzzxz(x,y,z)
    arr[(1, 2, 2, 1, 0)] = Tyzzyx(x,y,z)
    arr[(1, 2, 2, 1, 1)] = Tyzzyy(x,y,z)
    arr[(1, 2, 2, 1, 2)] = Tyzzyz(x,y,z)
    arr[(1, 2, 2, 2, 0)] = Tyzzzx(x,y,z)
    arr[(1, 2, 2, 2, 1)] = Tyzzzy(x,y,z)
    arr[(1, 2, 2, 2, 2)] = Tyzzzz(x,y,z)
    arr[(2, 0, 0, 0, 0)] = Tzxxxx(x,y,z)
    arr[(2, 0, 0, 0, 1)] = Tzxxxy(x,y,z)
    arr[(2, 0, 0, 0, 2)] = Tzxxxz(x,y,z)
    arr[(2, 0, 0, 1, 0)] = Tzxxyx(x,y,z)
    arr[(2, 0, 0, 1, 1)] = Tzxxyy(x,y,z)
    arr[(2, 0, 0, 1, 2)] = Tzxxyz(x,y,z)
    arr[(2, 0, 0, 2, 0)] = Tzxxzx(x,y,z)
    arr[(2, 0, 0, 2, 1)] = Tzxxzy(x,y,z)
    arr[(2, 0, 0, 2, 2)] = Tzxxzz(x,y,z)
    arr[(2, 0, 1, 0, 0)] = Tzxyxx(x,y,z)
    arr[(2, 0, 1, 0, 1)] = Tzxyxy(x,y,z)
    arr[(2, 0, 1, 0, 2)] = Tzxyxz(x,y,z)
    arr[(2, 0, 1, 1, 0)] = Tzxyyx(x,y,z)
    arr[(2, 0, 1, 1, 1)] = Tzxyyy(x,y,z)
    arr[(2, 0, 1, 1, 2)] = Tzxyyz(x,y,z)
    arr[(2, 0, 1, 2, 0)] = Tzxyzx(x,y,z)
    arr[(2, 0, 1, 2, 1)] = Tzxyzy(x,y,z)
    arr[(2, 0, 1, 2, 2)] = Tzxyzz(x,y,z)
    arr[(2, 0, 2, 0, 0)] = Tzxzxx(x,y,z)
    arr[(2, 0, 2, 0, 1)] = Tzxzxy(x,y,z)
    arr[(2, 0, 2, 0, 2)] = Tzxzxz(x,y,z)
    arr[(2, 0, 2, 1, 0)] = Tzxzyx(x,y,z)
    arr[(2, 0, 2, 1, 1)] = Tzxzyy(x,y,z)
    arr[(2, 0, 2, 1, 2)] = Tzxzyz(x,y,z)
    arr[(2, 0, 2, 2, 0)] = Tzxzzx(x,y,z)
    arr[(2, 0, 2, 2, 1)] = Tzxzzy(x,y,z)
    arr[(2, 0, 2, 2, 2)] = Tzxzzz(x,y,z)
    arr[(2, 1, 0, 0, 0)] = Tzyxxx(x,y,z)
    arr[(2, 1, 0, 0, 1)] = Tzyxxy(x,y,z)
    arr[(2, 1, 0, 0, 2)] = Tzyxxz(x,y,z)
    arr[(2, 1, 0, 1, 0)] = Tzyxyx(x,y,z)
    arr[(2, 1, 0, 1, 1)] = Tzyxyy(x,y,z)
    arr[(2, 1, 0, 1, 2)] = Tzyxyz(x,y,z)
    arr[(2, 1, 0, 2, 0)] = Tzyxzx(x,y,z)
    arr[(2, 1, 0, 2, 1)] = Tzyxzy(x,y,z)
    arr[(2, 1, 0, 2, 2)] = Tzyxzz(x,y,z)
    arr[(2, 1, 1, 0, 0)] = Tzyyxx(x,y,z)
    arr[(2, 1, 1, 0, 1)] = Tzyyxy(x,y,z)
    arr[(2, 1, 1, 0, 2)] = Tzyyxz(x,y,z)
    arr[(2, 1, 1, 1, 0)] = Tzyyyx(x,y,z)
    arr[(2, 1, 1, 1, 1)] = Tzyyyy(x,y,z)
    arr[(2, 1, 1, 1, 2)] = Tzyyyz(x,y,z)
    arr[(2, 1, 1, 2, 0)] = Tzyyzx(x,y,z)
    arr[(2, 1, 1, 2, 1)] = Tzyyzy(x,y,z)
    arr[(2, 1, 1, 2, 2)] = Tzyyzz(x,y,z)
    arr[(2, 1, 2, 0, 0)] = Tzyzxx(x,y,z)
    arr[(2, 1, 2, 0, 1)] = Tzyzxy(x,y,z)
    arr[(2, 1, 2, 0, 2)] = Tzyzxz(x,y,z)
    arr[(2, 1, 2, 1, 0)] = Tzyzyx(x,y,z)
    arr[(2, 1, 2, 1, 1)] = Tzyzyy(x,y,z)
    arr[(2, 1, 2, 1, 2)] = Tzyzyz(x,y,z)
    arr[(2, 1, 2, 2, 0)] = Tzyzzx(x,y,z)
    arr[(2, 1, 2, 2, 1)] = Tzyzzy(x,y,z)
    arr[(2, 1, 2, 2, 2)] = Tzyzzz(x,y,z)
    arr[(2, 2, 0, 0, 0)] = Tzzxxx(x,y,z)
    arr[(2, 2, 0, 0, 1)] = Tzzxxy(x,y,z)
    arr[(2, 2, 0, 0, 2)] = Tzzxxz(x,y,z)
    arr[(2, 2, 0, 1, 0)] = Tzzxyx(x,y,z)
    arr[(2, 2, 0, 1, 1)] = Tzzxyy(x,y,z)
    arr[(2, 2, 0, 1, 2)] = Tzzxyz(x,y,z)
    arr[(2, 2, 0, 2, 0)] = Tzzxzx(x,y,z)
    arr[(2, 2, 0, 2, 1)] = Tzzxzy(x,y,z)
    arr[(2, 2, 0, 2, 2)] = Tzzxzz(x,y,z)
    arr[(2, 2, 1, 0, 0)] = Tzzyxx(x,y,z)
    arr[(2, 2, 1, 0, 1)] = Tzzyxy(x,y,z)
    arr[(2, 2, 1, 0, 2)] = Tzzyxz(x,y,z)
    arr[(2, 2, 1, 1, 0)] = Tzzyyx(x,y,z)
    arr[(2, 2, 1, 1, 1)] = Tzzyyy(x,y,z)
    arr[(2, 2, 1, 1, 2)] = Tzzyyz(x,y,z)
    arr[(2, 2, 1, 2, 0)] = Tzzyzx(x,y,z)
    arr[(2, 2, 1, 2, 1)] = Tzzyzy(x,y,z)
    arr[(2, 2, 1, 2, 2)] = Tzzyzz(x,y,z)
    arr[(2, 2, 2, 0, 0)] = Tzzzxx(x,y,z)
    arr[(2, 2, 2, 0, 1)] = Tzzzxy(x,y,z)
    arr[(2, 2, 2, 0, 2)] = Tzzzxz(x,y,z)
    arr[(2, 2, 2, 1, 0)] = Tzzzyx(x,y,z)
    arr[(2, 2, 2, 1, 1)] = Tzzzyy(x,y,z)
    arr[(2, 2, 2, 1, 2)] = Tzzzyz(x,y,z)
    arr[(2, 2, 2, 2, 0)] = Tzzzzx(x,y,z)
    arr[(2, 2, 2, 2, 1)] = Tzzzzy(x,y,z)
    arr[(2, 2, 2, 2, 2)] = Tzzzzz(x,y,z)
    return arr
T = [T0, T1, T2, T3, T4, T5, ]
