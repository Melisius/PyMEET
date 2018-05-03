from pyscf import gto, scf
import numpy as np
from pyscf.lib import misc
misc.num_threads(n=1)

from pymeet.ESP import calculate_ESP
from pymeet.Mfit import Mfit
from pymeet.COM import calc_center_of_mass

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
    
    
def test_charge_A_and_B():
    points = np.array([[1.0, 1.0, 1.0]])
    ESP = np.array([[0.5, 1.0, 1.0, 1.0]])
    points = points.astype(float)
    molecule = np.array([["H","0.10583545","0.158753175","0.264588625"],["Li","0.2116709","0.264588625","0.317506349"]])
    A = np.array([[0.724637681,0.97009685],
    [0.97009685,1.298701299]])
    B = np.array([0.425628265,
    0.569802882])
    
    Q = Mfit(molecule, points, "STO3G", 0.0, multipole_order=0, precalculated_ESP=ESP)
    
    assert np.max(np.abs(A - Q.A_matrix)) < 10**-8
    assert np.max(np.abs(B - Q.B_vector)) < 10**-8
    
    
def test_mu_A_and_B():
    points = np.array([[1.0, 1.0, 1.0]])
    ESP = np.array([[0.5, 1.0, 1.0, 1.0]])
    points = points.astype(float)
    molecule = np.array([["H","0.10583545","0.158753175","0.264588625"],["Li","0.2116709","0.264588625","0.317506349"]])
    A = np.array([[0.724637681,0.97009685,0.420079815,0.755919623,0.367569838,0.629933019,0.262549884,0.503946415],
    [0.97009685,1.298701299,0.562374985,1.011975038,0.492078112,0.843312532,0.351484366,0.674650025],
    [0.420079815,0.562374985,0.243524531,0.438214274,0.213083964,0.365178562,0.152202832,0.292142849],
    [0.755919623,1.011975038,0.438214274,0.788551978,0.38343749,0.657126648,0.2738839219,0.525701318],
    [0.367569838,0.492078112,0.213083964,0.38343749,0.186448469,0.319531242,0.133177478,0.255624993],
    [0.629933019,0.843312532,0.365178562,0.657126648,0.319531242,0.54760554,0.228236601,0.438084432],
    [0.262549884,0.351484366,0.152202832,0.273883921,0.133177478,0.228236601,0.09512677,0.182589281],
    [0.503946415,0.674650025,0.292142849,0.525701318,0.255624993,0.438084432,0.182589281,0.350467546]])
    B = np.array([0.425628265,
    0.569802882,
    0.246741023,
    0.444002246,
    0.215898395,
    0.370001872,
    0.15421314,
    0.296001497])
    
    Q = Mfit(molecule, points, "STO3G", 0.0, multipole_order=1, precalculated_ESP=ESP)
    
    assert np.max(np.abs(A - Q.A_matrix)) < 10**-8
    assert np.max(np.abs(B - Q.B_vector)) < 10**-8
    
    
def test_some_theta_A_and_B():
    points = np.array([[1.0, 1.0, 1.0]])
    ESP = np.array([[0.5, 1.0, 1.0, 1.0]])
    points = points.astype(float)
    molecule = np.array([["H","0.10583545","0.158753175","0.264588625"],["Li","0.2116709","0.264588625","0.317506349"]])
    Q = Mfit(molecule, points, "STO3G", 0.0, multipole_order=2, precalculated_ESP=ESP)
    
    assert np.abs(Q.A_matrix[8,8] - 0.25*0.0818397918186485) < 10**-8
    assert np.abs(Q.A_matrix[8,9] - 0.25*0.19795111202344) < 10**-8
    assert np.abs(Q.A_matrix[8,10] - 0.25*0.0716098178413174) < 10**-8
    assert np.abs(Q.A_matrix[8,11] - 0.25*0.164959260019533) < 10**-8
    assert np.abs(Q.A_matrix[8,12] - 0.25*0.0511498698866553) < 10**-8
    assert np.abs(Q.A_matrix[8,13] - 0.25*0.131967408015627) < 10**-8
    
    
def test_charge_constraint():
    points = np.genfromtxt("data/testfiles/surface_points.txt", delimiter=" ")
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
    C = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=0, charge_constraint=0)
    C.fit_multipoles()
    
    assert np.sum(C.fitted_moments[0:3]) < 10**-12
    
    
def test_dipole_constraint():
    points = np.genfromtxt("data/testfiles/surface_points.txt", delimiter=" ")
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
    C = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=1, charge_constraint=0, dipole_constraint=np.array([1.0, 1.0, 1.0]))
    C.fit_multipoles()
    COM = C.Center_of_Mass
    dipole = np.array([0.0, 0.0, 0.0])
    dipole[0] += np.sum(C.fitted_moments[3:6])
    dipole[1] += np.sum(C.fitted_moments[6:9])
    dipole[2] += np.sum(C.fitted_moments[9:12])
    R = np.zeros((3,3))
    R[:,0] = C.molecule_coords[:,0] - COM[0]
    R[:,1] = C.molecule_coords[:,1] - COM[1]
    R[:,2] = C.molecule_coords[:,2] - COM[2]
    dipole[0] += C.fitted_moments[0]*R[0,0] + C.fitted_moments[1]*R[1,0] + C.fitted_moments[2]*R[2,0]
    dipole[1] += C.fitted_moments[0]*R[0,1] + C.fitted_moments[1]*R[1,1] + C.fitted_moments[2]*R[2,1]
    dipole[2] += C.fitted_moments[0]*R[0,2] + C.fitted_moments[1]*R[1,2] + C.fitted_moments[2]*R[2,2]
    assert np.abs(dipole[0] - 1.0) < 10**-12
    assert np.abs(dipole[1] - 1.0) < 10**-12
    assert np.abs(dipole[2] - 1.0) < 10**-12
    
    C = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=1, charge_constraint=0, dipole_constraint=np.array([1.0, 1.0, 1.0]))
    C.fit_multipoles()
    dipole = np.array([0.0, 0.0, 0.0])
    dipole[0] += np.sum(C.fitted_moments[3:6])
    dipole[1] += np.sum(C.fitted_moments[6:9])
    dipole[2] += np.sum(C.fitted_moments[9:12])
    R = np.zeros((3,3))
    R[:,0] = C.molecule_coords[:,0] - COM[0]
    R[:,1] = C.molecule_coords[:,1] - COM[1]
    R[:,2] = C.molecule_coords[:,2] - COM[2]
    dipole[0] += C.fitted_moments[0]*R[0,0] + C.fitted_moments[1]*R[1,0] + C.fitted_moments[2]*R[2,0]
    dipole[1] += C.fitted_moments[0]*R[0,1] + C.fitted_moments[1]*R[1,1] + C.fitted_moments[2]*R[2,1]
    dipole[2] += C.fitted_moments[0]*R[0,2] + C.fitted_moments[1]*R[1,2] + C.fitted_moments[2]*R[2,2]
    assert np.abs(dipole[0] - 1.0) < 10**-12
    assert np.abs(dipole[1] - 1.0) < 10**-12
    assert np.abs(dipole[2] - 1.0) < 10**-12
    assert np.abs(np.sum(C.fitted_moments[0:3])) < 10**-12


def test_by_RMSD():
    points = np.genfromtxt("data/testfiles/surface_points.txt", delimiter=" ")
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
    C = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=0)
    C.fit_multipoles()
    q_rmsd = C.get_RMSD()
    C = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=1)
    C.fit_multipoles()
    mu_rmsd = C.get_RMSD()
    C = Mfit(molecule, points, "STO3G", m.make_rdm1(), multipole_order=2)
    C.fit_multipoles()
    theta_rmsd = C.get_RMSD()
    assert q_rmsd > mu_rmsd
    assert mu_rmsd > theta_rmsd
    

def test_center_of_mass():
    xyz = np.array([[0.000000000000, 0.000000000000, 0.000000000000],
    [0.000000000000, 0.000000000000, 2.845112131228],
    [1.899115961744, 0.000000000000, 4.139062527233],
    [-1.894048308506, 0.000000000000, 3.747688672216],
    [1.942500819960, 0.000000000000,-0.701145981971],
    [-1.007295466862,-1.669971842687,-0.705916966833],
    [-1.007295466862, 1.669971842687,-0.705916966833]])
    elements = np.array(["C","C","O","H","H","H","H"])
    com = calc_center_of_mass(elements, xyz)
    assert np.abs(com[0] - 0.64494926) < 10**-3
    assert np.abs(com[1] - 0.00000000) < 10**-3
    assert np.abs(com[2] - 2.31663792) <10**-3
    
    xyz = np.array([[0.000000000000, 0.000000000000, 0.000000000000],
    [0.000000000000, 0.000000000000, 2.616448463377],
    [2.265910476936, 0.000000000000, 3.924672487195],
    [4.531821084796, 0.000000000000, 2.616448387788],
    [4.531821084796, 0.000000000000,-0.000000132281],
    [2.265910689687, 0.000000000000,-1.308224146651],
    [2.265910689687,-0.000000000000,-3.334218200677],
    [6.286383272575, 0.000000000000,-1.012997083705],
    [6.286383239844, 0.000000000000, 3.629445320315],
    [2.265910476936, 0.000000000000, 5.950666541222],
    [-1.754562187779, 0.000000000000, 3.629445414801],
    [-1.754562155048,-0.000000000000,-1.012996932527]])
    elements = np.array(["C","C","C","C","C","C","H","H","H","H","H","H"])
    com = calc_center_of_mass(elements, xyz)
    assert np.abs(com[0] - 2.26591056) < 10**-3
    assert np.abs(com[1] - 0.00000000) < 10**-3
    assert np.abs(com[2] - 1.30822418) <10**-3