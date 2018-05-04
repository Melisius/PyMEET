from numpy import nan
from numba import jit
import numpy as np

name2number = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
        "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
        "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
        "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
        "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76,
        "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
        "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg":
        106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}

number2name = np.array(["nope", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"])


vdw_radii = np.array([nan, 2.26767118629, 2.64561638401, 3.43930129921, 2.89128076253, 3.62827389807, 3.21253418058, 2.9290752823, 2.87238350264, 2.77789720321,
    2.91017802241, 4.28967799407, 3.26922596024, 3.47709581899, 3.96842457602, 3.40150677944, 3.40150677944, 3.30702048001, 3.55268485853, 5.19674646859,
    4.36526703362, 3.9873218359, nan, nan, nan, nan, nan, nan, 3.08025336138, 2.64561638401, 2.62671912412, 3.53378759864, 3.9873218359, 3.49599307887,
    3.5904793783, 3.49599307887, 3.81724649693, 5.72586974539, 4.70541771156, nan, nan, nan, nan, nan, nan, nan, 3.08025336138, 3.25032870036, 2.98576706195,
    3.64717115796, 4.10070539522, 3.89283553647, 3.89283553647, 3.74165745739, 4.08180813533, 6.48176014083, 5.06446564939, nan, nan, nan, nan, nan, nan, nan,
    nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 3.30702048001, 3.13694514104, 2.9290752823, 3.70386293761, 3.81724649693,
    3.91173279636, 3.7227601975, 3.81724649693, 4.15739717487, 6.57624644025, 5.34792454768, nan, nan, nan, 3.51489033876, nan, nan, nan, nan, nan, nan, nan,
    nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], dtype=np.float64)

import numpy as np
from numba import jit, boolean

@jit('float64(float64[:], float64[:])', nopython=True, cache=True)
def dist(x,y):
    """
    Compute distance between vectors x,y
    Assumes that x,y have the same length
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i])**2
    return np.sqrt(result)


@jit(nopython=True, cache=True)
def compute_vdW_surface(atomic_charges, coordinates, surface_point_density=2.0, surface_vdW_scale=1.4):
    """
    Generates apparent uniformly spaced points on a vdw_radii
    surface of a molecule.

    vdw_radii   = van der Waals radius of atoms
    points      = number of points on a sphere around each atom IN ATOMIC UNITS
    grid        = output points in x, y, z
    idx         = used to keep track of index in grid, when generating
                  initial points
    density     = points per area on a surface
    chkrm       = (checkremove) used to keep track in index when
                  removing points
    """
    natoms = coordinates.shape[0]
    points = np.zeros(natoms, dtype=np.int64)
    for i in range(natoms):
        #        area of sphere is               4*pi    *        r**2
        points[i] = np.int(surface_point_density*4*np.pi*(surface_vdW_scale*vdw_radii[atomic_charges[i]])**2)
    # grid = [x, y, z]
    grid = np.zeros((np.sum(points), 3), dtype=np.float64)
    idx = 0
    for i in range(natoms):
        N = points[i]
        #Saff & Kuijlaars algorithm
        for k in range(N):
            h = -1.0 +2.0*k/(N-1)
            theta = np.arccos(h)
            if k == 0 or k == (N-1):
                phi = 0.0
            else:
                #phi_k  phi_{k-1}
                phi = ((phi + 3.6/np.sqrt(N*(1-h**2)))) % (2*np.pi)
            x = surface_vdW_scale*vdw_radii[atomic_charges[i]]*np.cos(phi)*np.sin(theta)
            y = surface_vdW_scale*vdw_radii[atomic_charges[i]]*np.sin(phi)*np.sin(theta)
            z = surface_vdW_scale*vdw_radii[atomic_charges[i]]*np.cos(theta)
            grid[idx, 0] = x + coordinates[i,0]
            grid[idx, 1] = y + coordinates[i,1]
            grid[idx, 2] = z + coordinates[i,2]
            idx += 1



    #This is the distance points have to be apart
    #since they are from the same atom
    grid_spacing = dist(grid[1,:], grid[2,:])

    #Remove overlap all points to close to any atom
    not_near_atom = np.ones(grid.shape[0], dtype=boolean)
    for i in range(natoms):
        for j in range(grid.shape[0]):
            r = dist(grid[j,:], coordinates[i,:])
            if r < surface_vdW_scale*0.99*vdw_radii[atomic_charges[i]]:
                not_near_atom[j] = False
    grid = grid[not_near_atom]

    # Double loop over grid to remove close lying points
    not_overlapping = np.ones(grid.shape[0], dtype=boolean)
    for i in range(grid.shape[0]):
        for j in range(i+1, grid.shape[0]):
            if (not not_overlapping[j]): continue #already marked for removal
            r = dist(grid[i,:], grid[j,:])
            if 0.80 * grid_spacing > r:
                not_overlapping[j] = False
    grid = grid[not_overlapping]
    return grid


def compute_grid(molecule, rmin=1.4, rmax=2.0, pointdensity=1.0, nsurfaces=2):
    """
    molecule coordinates are in Angstrom
    """
    radii = np.linspace(rmin, rmax, nsurfaces)
    surfaces = []
    atomic_charges = [name2number[el] for el in molecule[:,0]]
    # conversion to A.U.
    coordinates = 1.889725989 * np.array(molecule[:,1:], dtype=np.float64)
    for r in radii:
        surfaces.append(compute_vdW_surface(atomic_charges,
                                            coordinates,
                                            surface_point_density=pointdensity,
                                            surface_vdW_scale=r))
    grid        = np.concatenate(surfaces)
    return grid
