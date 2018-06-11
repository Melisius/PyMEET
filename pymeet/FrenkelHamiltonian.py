import itertools
from string import ascii_lowercase as abc

from pymeet.tensors import tensors_batched
from pymeet.tensors import functions
import numpy as np

angstrom_to_au = 1.889725989


def compute_interaction_energy(k, l, mul1, mul2, r12):
    rank = k + l
    Tkl = tensors_batched.T[rank](r12, 0)

#     print(mul1.shape, mul2.shape, Tkl.shape)

    if k == 0:
        mul1 = mul1[0]
    if l == 0:
        mul2 = mul2[0]
    Tkl = Tkl[0]

    factor = (-1)**rank / functions.factorial[rank]
    signature = '{},{},{}->'.format(abc[:k], abc[:k + l], abc[k:k + l])
#     print(signature)
    res = factor * np.einsum(signature, mul1, Tkl, mul2)
    return res


class FrenkelHamiltonian:
    """docstring for FrenkelHamiltonian."""

    def __init__(self, chromophores):
        self.chromophores = chromophores
        self.matrix = None
        self.evals, self.evecs = None, None

    def construct(self):
        # get possible combinations of chromophores
        chromophore_combinations = \
            list(itertools.combinations(range(len(self.chromophores)), 2))

        n = 0
        n_states_chromophore_start = []
        all_exc_energies = np.array([])
        for c in self.chromophores:
            n_states_chromophore_start.append(n)
            n += c.n_states
            all_exc_energies = np.concatenate((all_exc_energies, c.etenergies))
        self.matrix = np.zeros(shape=(n, n))
        np.fill_diagonal(self.matrix, all_exc_energies)
        print("Shape of H: ", self.matrix.shape)

        for combination in chromophore_combinations:
            c1, c2 = self.chromophores[combination[0]], self.chromophores[combination[1]]

            # J_pq
            pq_combinations = list(itertools.product(range(c1.n_states),
                                                     range(c2.n_states)))

            # compute J_1
            J_1 = np.zeros(shape=(c1.n_states, c2.n_states))
            J_0 = np.zeros(shape=(c1.n_states, c2.n_states))
            J_1_test = np.zeros(shape=(c1.n_states, c2.n_states))
            for p, q in pq_combinations:
                print("state c1: {}, state c2: {}".format(p, q))
                J_1[p, q] = c1.ind_moments_per_state[p] @ c2.fields_per_state[q]
                J_1_test[p, q] = c1.fields_per_state[p] @ c2.ind_moments_per_state[q]

                multipoles1 = c1.multipoles_per_state[p]
                multipoles2 = c2.multipoles_per_state[q]
                for m1 in range(multipoles1.max_k + 1):
                    for m2 in range(multipoles2.max_k + 1):
                        print(m1, m2)
                        for idx1, at1 in enumerate(c1.coords * angstrom_to_au):
                            for idx2, at2 in enumerate(c2.coords * angstrom_to_au):
                                r12 = (at2 - at1).reshape((1, 3))
                                en = compute_interaction_energy(m1, m2, multipoles1.Mlist[m1][idx1],
                                                                multipoles2.Mlist[m2][idx2],
                                                                -r12)
        #                         print(en)
            assert np.allclose(J_1, J_1_test) is True
            i = n_states_chromophore_start[combination[0]]
            j = n_states_chromophore_start[combination[1]]
            coupling_block = J_0 + J_1
            self.matrix[i:i + c1.n_states,
                        j:j + c2.n_states] = coupling_block
            self.matrix[j:j + c2.n_states,
                        i:i + c1.n_states] = coupling_block.transpose(1, 0)
        print(self.matrix)

    def diagonalize(self):
        self.evals, self.evecs = np.linalg.eigh(self.matrix)
