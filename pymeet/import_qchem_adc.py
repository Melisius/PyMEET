import glob

from cclib.parser import QChem
from cclib.parser.utils import PeriodicTable
from cclib.parser.utils import convertor
import numpy as np

from .Chromophore import Chromophore


def get_chromophore_info(calcdir, basis="None"):
    """
    calcdir: directory the output file resides in
    """
    outfile = glob.glob("{}/*.out".format(calcdir))[0]
    results = QChem(outfile).parse()
    n_states = len(results.etenergies)
    if False in results.etconv:
        print("Not converged!")
    # Reads in the TXT files with the transition density matrix
    ptr_files = sorted(glob.glob("{}/*.txt".format(calcdir)))[:n_states]
    ptr = []
    periodicT = PeriodicTable()
    for ptr_f in ptr_files:
        ptr_mat = np.loadtxt(ptr_f)
        assert ptr_mat.shape[0] == ptr_mat.shape[1]
        assert ptr_mat.ndim == 2
        ptr.append(ptr_mat)
    chrom = Chromophore(n_states, basis,
                        results.atomcoords[0],
                        [periodicT.element[x] for x in results.atomnos],
                        results.charge,
                        convertor(results.etenergies, "eV", "hartree"),
                        results.etoscs,
                        ptr, results.ettransdipmoms)
    return chrom
