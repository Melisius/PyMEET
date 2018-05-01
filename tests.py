from pyscf import gto, scf
import numpy as np
from pyscf.lib import misc
misc.num_threads(n=1)

from pymeet.ESP import calculate_ESP
from pymeet.Mfit import Mfit

def test_ESP():
    points = np.genfromtxt("data/testfiles/surface_points.txt", delimiter=" ")
    check = np.genfromtxt("data/testfiles/ESP_reference.txt", delimiter=" ")
    mol = gto.Mole()
    mol.atom = '''H 0.866811829 0.601435780 0.000000000; O 0.000000000 -0.075791844 0.000000000; H -0.866811829 0.601435780 0.000000000'''
    mol.basis = 'STO3G'
    mol.build()
    m = scf.RHF(mol)
    m.kernel()
    
    temp = open("data/testfiles/H2O.xyz").readlines()
    molecule = []
    for i in range(2, len(temp)):
        molecule.append(temp[i].split())
    calc = calculate_ESP(points, molecule, "STO3G", m.make_rdm1())
    assert np.max(np.abs(calc[:,0]-check)) < 10**-8
    
    
def test_charge_fitting():
    points = np.genfromtxt("data/testfiles/surface_points.txt", delimiter=" ")
    #check = np.genfromtxt("data/testfiles/ESP_reference.txt", delimiter=" ")
    mol = gto.Mole()
    mol.atom = '''H 0.866811829 0.601435780 0.000000000; O 0.000000000 -0.075791844 0.000000000; H -0.866811829 0.601435780 0.000000000'''
    mol.basis = 'STO3G'
    mol.build()
    m = scf.RHF(mol)
    m.kernel()
    
    temp = open("data/testfiles/H2O.xyz").readlines()
    molecule = []
    for i in range(2, len(temp)):
        molecule.append(temp[i].split())
    fit_test = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=0)
    print(fit_test.fit_multipoles())
    
    
test_charge_fitting()  