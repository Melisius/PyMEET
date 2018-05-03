import numpy as np

class Potential:
    def __init__(self, n_atoms, M0, M1, M2):
        self.M0 = np.array(M0, dtype=np.float64)
        self.M1 = np.array(M1, dtype=np.float64)
        self.M2 = np.array(M2, dtype=np.float64)
        self.n_atoms = n_atoms
        assert self.M0.size == 1*n_atoms
        assert self.M1.size == 3*n_atoms
        assert self.M1.size == 6*n_atoms
