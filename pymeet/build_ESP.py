from pyscf import gto
import numpy as np
from pyscf.lib import misc
misc.num_threads(n=1)


def calculate_Nuclear_Potential(PySCF_molecule, point_xyz):
    """
    Calculates the nuclear potential in a point.
    
    Input : PySCF_molecule, PySCF molecule object.
          : point_xyz, x, y and z coordinates of point.
          
    Output : Nuclear_Potential, nuclear potential in point.
    """
    Nuclear_Potential = 0.0
    for i in range(PySCF_molecule.natm):
       xyz = PySCF_molecule.atom_coord(i)
       charge = PySCF_molecule.atom_charge(i)
       R = xyz - point_xyz
       Nuclear_Potential += charge/np.dot(R,R)**0.5
    return Nuclear_Potential


def calculate_ESP(surface_points, molecule_xyz_file, basis_set, density_matrix):
    """
    Calculated the electronic and nuclear ESP in a set of points.
    
    Input : surface_points, the points for which the ESP is evaluated in au.
          : molecule_xyz_file, path to xyz-file of molecule in angstrom.
          : basis_set, basis set used to construct density matrix.
          : density matrix.
          
    Output : surface_ESP, array of ESP values [ESP_value, x, y, z]
    """
    surface_ESP = np.zeros((len(surface_points),4))
    surface_ESP[:,1:] = surface_points
    molecule_xyz = open(molecule_xyz_file, "r").readlines()
    
    molecule = []
    atoms = ''
    for i in range(2, len(molecule_xyz)):
        atom = molecule_xyz[i].split()
        molecule.append([atom[0],(float(atom[1]),float(atom[2]),float(atom[3]))])
        atoms = atoms+atom[0]
    
    mol = gto.Mole()
    mol.atom = molecule
    mol.basis = basis_set
    mol.build()
    
    for i in range(0, len(surface_ESP)):
        mol.set_rinv_origin((surface_ESP[i,1:]))
        rinv_integral = mol.intor("int1e_rinv_sph")
        Electronic_ESP = np.einsum("jk,jk->", density_matrix, rinv_integral)
        Nuclear_ESP = calculate_Nuclear_Potential(mol, surface_ESP[i,1:])
        surface_ESP[i,0] = Nuclear_ESP - 2*Electronic_ESP
        
    return surface_ESP