import numpy as np

class Multipoles:
    def __init__(self, n_atoms, max_k, M0=None, M1=None, M2=None):
        self.M0 = M0
        self.M1 = M1
        self.M2 = M2
        self.max_k = max_k
        self.n_atoms = n_atoms
