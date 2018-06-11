from . import tensors
import numpy as np
from .decorators import memo1
from numba import jit


@jit(nopython=True, cache=True)
def T0(Rab, idx):
    natoms = Rab.shape[0]
    out = np.zeros((natoms))
    for k in range(natoms):
        out[k] = 1.0/np.sqrt((Rab[k,0]**2 + Rab[k,1]**2 + Rab[k,2]**2))
    return out

@jit(nopython=True, cache=True)
def T1(Rab, idx):
    natoms = Rab.shape[0]
    out = np.zeros((natoms,3))
    for k in range(natoms):
        out[k] = tensors.T1(Rab[k,0], Rab[k,1], Rab[k,2])
    return out


@jit(nopython=True, cache=True)
def T2(Rab, idx):
    natoms = Rab.shape[0]
    out = np.zeros((natoms,3,3))
    for k in range(natoms):
        out[k] = tensors.T2(Rab[k,0], Rab[k,1], Rab[k,2])
    return out


@jit(nopython=True, cache=True)
def T3(Rab, idx):
    natoms = Rab.shape[0]
    out = np.zeros((natoms,3,3,3))
    for k in range(natoms):
        out[k] = tensors.T3(Rab[k,0], Rab[k,1], Rab[k,2])
    return out


@jit(nopython=True, cache=True)
def T4(Rab, idx):
    natoms = Rab.shape[0]
    out = np.zeros((natoms,3,3,3,3))
    for k in range(natoms):
        out[k] = tensors.T4(Rab[k,0], Rab[k,1], Rab[k,2])
    return out


@jit(nopython=True, cache=True)
def T5(Rab, idx):
    natoms = Rab.shape[0]
    out = np.zeros((natoms,3,3,3,3,3))
    for k in range(natoms):
        out[k] = tensors.T5(Rab[k,0], Rab[k,1], Rab[k,2])
    return out

T = [T0, T1, T2, T3, T4, T5, ]
T_memo = [memo1(T0), memo1(T1), memo1(T2), memo1(T3), memo1(T4), memo1(T5), ]
