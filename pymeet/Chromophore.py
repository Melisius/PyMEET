import numpy as np
from pepytools import Potential

from pymeet.Multipoles import Multipoles

class Chromophore:
    """
    Container class for EET calculations

    Contains all the important information
    """

    def __init__(self, n_states, basis_set, coords,
                 elements, charge, etenergies, etoscs, ptr_mats, tr_dipmoms):
        """

        Parameters
        ----------
        n_states: int
            number of excited states
        basis_set: str
            basis set name
        coords: np.ndarray
            atomic coordinates in Angstrom
        elements: list
            list of elements
        charge: int
            total charge of the chromophore
        etenergies: np.array
            list of excitation energies in eV
        etoscs: np.array
            list of oscillator strengths
        ptr_mats: list of np.ndarrays
            list of transition density matrices, length equal to n_states
        """
        self.n_states = n_states
        self.basis_set = basis_set
        self.coords = coords
        self.elements = elements
        self.charge = charge
        self.etenergies = etenergies
        self.etoscs = etoscs
        assert len(ptr_mats) == n_states
        self.ptr_mats = ptr_mats
        self.tr_dipmoms = tr_dipmoms

        # list of multipoles per state
        self.multipoles_per_state = []
        self.potentials_per_state = []

        self.fields_per_state = []
        self.ind_moments_per_state = []

    def make_potentials(self):
        """
        Creates pepytools.Potential objects for every state

        """
        if len(self.multipoles_per_state) != self.n_states:
            raise Exception("Invalid number of multipole moments to create potential.", self.n_states, len(self.multipoles_per_state))
        self.potentials_per_state = []
        for state in range(self.n_states):
            max_k = self.multipoles_per_state[state].max_k
            m0 = [list(x) for x in self.multipoles_per_state[state].M0]
            if max_k == 0:
                m1 = None
                m2 = None
            elif max_k == 1:
                m1 = [list(x) for x in self.multipoles_per_state[state].M1]
                m2 = None
            elif max_k == 2:
                m1 = [list(x) for x in self.multipoles_per_state[state].M1]
                m2 = [list(x) for x in self.multipoles_per_state[state].M2]
            else:
                m1 = None
                m2 = None
            moment_list = [x for x in [m0, m1, m2] if x is not None]
            # pepytools wants coordinates in AU
            p = Potential.from_multipoles(1.889725989 * self.coords, moment_list, max_k=max_k)
            self.potentials_per_state.append(p)

    def __str__(self):
        return """
        nstates: {}
        number of atoms: {}
        elements: {}
        coords:
        {}
        excitation energies: {}
        tdm: {}
        f: {}
        """.format(self.n_states, len(self.elements), self.elements, self.coords,
                  self.etenergies, self.tr_dipmoms, self.etoscs)
