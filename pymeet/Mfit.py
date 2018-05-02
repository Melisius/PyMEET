import numpy as np
from pymeet.VectorConstructor import *
from pymeet.ESP import calculate_ESP
from pymeet.COM import calc_center_of_mass


class Mfit():
    def __init__(self, molecule, surface, basis_set, density_matrix, multipole_order=2, charge_constraint=None, dipole_constraint=np.array(["None"]), precalculated_ESP=np.array([["None"]])):
        """
        Main part of PyMEET.
        Fits multipole moments to QM calculated ESPs.
        
        Input : molecule, xyz of molecule in angstrom. In the format [atom, x, y, z] all string.
              : surface, point for where the ESP is to be calculated.
              : basis_set, basis set to be used.
              : denisty_matrix, density matrix.
              : multipole_order, maximum order of multipoles to be fitted. Default = 2.
              : charge_constraint, total charge of the molecule. Default = None.
              : dipole_constraint, molecular dipole molement given as [mu_x, mu_y, mu_z]. Default = None.
              : precalculated_ESP, specify precalculated ESP here. [value, x, y, z]. 
                                   the x, y, z should be identical to that of surface. 
              
        Output : call self.fit_multipoles()
        
        If parameters are changed call self.construct_A_and_B() to reconstruct needed matrix and vector.
        
        TODO : Remove use_precalculated_ESP, specifing precalculated_ESP should be enough.
             : Maybe change solver to SVD? All tests should pass when changing this
             : Be 100 % sure about pre-factors on quadropoles.
             : Option for traceless quadropoles?
        """
        self.__angstrom_to_au = 1.889725989
        self.multipole_order = multipole_order
        self.molecule_xyz = molecule
        self.elements = np.array(molecule, dtype=str)[:,0]
        self.molecule_coords = np.array(molecule, dtype=str)[:,1:].astype(float)*self.__angstrom_to_au
        self.charge_constraint = charge_constraint
        self.dipole_constraint = dipole_constraint
        self.number_atoms = len(molecule)
        self.surface_points = surface
        self.basis_set = basis_set
        self.density_matrix = density_matrix
        self.precalculated_ESP = precalculated_ESP
        
        self.Center_of_Mass = calc_center_of_mass(self.elements, self.molecule_coords)
        self.construct_A_and_B()


    def construct_A_and_B(self):
        problem_size = 0
        if self.multipole_order == 0:
            problem_size = self.number_atoms
        elif self.multipole_order == 1:
            problem_size = self.number_atoms*4
        elif self.multipole_order == 2:
            problem_size = self.number_atoms*10
        else:
            print("Cannot go further than multipole order two")
            quit()
    
        problem_size_constraints = 0
        if self.charge_constraint != None:
            problem_size_constraints += 1
        if self.dipole_constraint[0] != "None":
            problem_size_constraints += 3
        
        self.A_vector = np.zeros((problem_size,len(self.surface_points)))
        self.A_matrix = np.zeros((problem_size+problem_size_constraints,problem_size+problem_size_constraints))
        self.B_vector = np.zeros(problem_size+problem_size_constraints)
        
        if self.precalculated_ESP[0,0] != "None":
            self.ESP_values = self.precalculated_ESP
        else:
            if self.charge_constraint != None:
                self.ESP_values = calculate_ESP(self.surface_points, self.molecule_xyz, self.basis_set, self.density_matrix, charge=self.charge_constraint)
            else:
                self.ESP_values = calculate_ESP(self.surface_points, self.molecule_xyz, self.basis_set, self.density_matrix)
            
        # Construct A and B vector
        counter = 0
        if self.multipole_order >= 0:
            q_vector_A = Vector_q_A(self.molecule_coords, self.surface_points, self.ESP_values)
            q_vector_B = Vector_q_B(self.molecule_coords, self.surface_points, self.ESP_values)
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = q_vector_A
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = q_vector_B
            counter += 1
        if self.multipole_order >= 1:
            mu_vector_A = Vector_mu_A(self.molecule_coords, self.surface_points, self.ESP_values)
            mu_vector_B = Vector_mu_B(self.molecule_coords, self.surface_points, self.ESP_values)
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = mu_vector_A[0]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = mu_vector_B[0]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = mu_vector_A[1]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = mu_vector_B[1]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = mu_vector_A[2]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = mu_vector_B[2]
            counter += 1
        if self.multipole_order >= 2:
            theta_vector_A = Vector_theta_A(self.molecule_coords, self.surface_points, self.ESP_values)
            theta_vector_B = Vector_theta_B(self.molecule_coords, self.surface_points, self.ESP_values)
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = theta_vector_A[0]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = theta_vector_B[0]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = theta_vector_A[1]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = theta_vector_B[1]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = theta_vector_A[2]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = theta_vector_B[2]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = theta_vector_A[3]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = theta_vector_B[3]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = theta_vector_A[4]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = theta_vector_B[4]
            counter += 1
            self.A_vector[counter*self.number_atoms:(counter+1)*self.number_atoms,:] = theta_vector_A[5]
            self.B_vector[counter*self.number_atoms:(counter+1)*self.number_atoms] = theta_vector_B[5]
            counter += 1
        
        # Construct A matrix and B vector with constraints
        self.A_matrix[0:problem_size,0:problem_size] = np.dot(self.A_vector,np.transpose(self.A_vector))
        counter = 0
        if self.charge_constraint != None:
            # Charge contribution to charge constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,0:self.number_atoms] = self.A_matrix[0:self.number_atoms,problem_size+counter:problem_size+counter+1] = 1
            self.B_vector[problem_size+counter:problem_size+counter+1] = self.charge_constraint
            counter += 1
        if self.dipole_constraint[0] != "None":
            # Charge contribution to dipole_x constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,0:self.number_atoms] = np.array([self.molecule_coords[:,0] - self.Center_of_Mass[0]])
            self.A_matrix[0:self.number_atoms,problem_size+counter:problem_size+counter+1] = np.transpose(np.array([self.molecule_coords[:,0] - self.Center_of_Mass[0]]))
            # Dipole contribution to dipole_x constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,self.number_atoms:self.number_atoms*2] = self.A_matrix[self.number_atoms:self.number_atoms*2,problem_size+counter:problem_size+counter+1] = 1
            self.B_vector[problem_size+counter:problem_size+counter+1] = self.dipole_constraint[0]
            counter += 1
            # Charge contribution to dipole_y constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,0:self.number_atoms] = np.array([self.molecule_coords[:,1] - self.Center_of_Mass[1]])
            self.A_matrix[0:self.number_atoms,problem_size+counter:problem_size+counter+1] = np.transpose(np.array([self.molecule_coords[:,1] - self.Center_of_Mass[1]]))
            # Dipole contribution to dipole_y constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,self.number_atoms*2:self.number_atoms*3] = self.A_matrix[self.number_atoms*2:self.number_atoms*3,problem_size+counter:problem_size+counter+1] = 1
            self.B_vector[problem_size+counter:problem_size+counter+1] = self.dipole_constraint[1]
            counter += 1
            # Charge contribution to dipole_z constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,0:self.number_atoms] = np.array([self.molecule_coords[:,2] - self.Center_of_Mass[2]])
            self.A_matrix[0:self.number_atoms,problem_size+counter:problem_size+counter+1] = np.transpose(np.array([self.molecule_coords[:,2] - self.Center_of_Mass[2]]))
            # Dipole contribution to dipole_z constraint
            self.A_matrix[problem_size+counter:problem_size+counter+1,self.number_atoms*3:self.number_atoms*4] = self.A_matrix[self.number_atoms*3:self.number_atoms*4,problem_size+counter:problem_size+counter+1] = 1
            self.B_vector[problem_size+counter:problem_size+counter+1] = self.dipole_constraint[2]
            counter += 1
            
            
    def fit_multipoles(self):
        self.fitted_moments = np.linalg.solve(self.A_matrix, self.B_vector)
        return self.fitted_moments
       
        
    def get_RMSD(self):
        RMSD = 0
        for i in range(0, len(self.surface_points)):
            E = 0
            for j in range(0, len(self.molecule_coords)):
                if self.multipole_order >= 0:
                    R = self.molecule_coords[j,:] - self.surface_points[i,:]
                    E += self.fitted_moments[j]/np.dot(R,R)**0.5
                if self.multipole_order >= 1:
                    E += self.fitted_moments[j+len(self.molecule_coords)]*R[0]/(np.dot(R,R)**0.5)**3
                    E += self.fitted_moments[j+len(self.molecule_coords)*2]*R[1]/(np.dot(R,R)**0.5)**3
                    E += self.fitted_moments[j+len(self.molecule_coords)*3]*R[2]/(np.dot(R,R)**0.5)**3
                if self.multipole_order >= 2:
                    E += 0.5*self.fitted_moments[j+len(self.molecule_coords)*4]*R[0]*R[0]/(np.dot(R,R)**0.5)**5
                    E += 0.5*self.fitted_moments[j+len(self.molecule_coords)*5]*R[0]*R[1]/(np.dot(R,R)**0.5)**5
                    E += 0.5*self.fitted_moments[j+len(self.molecule_coords)*6]*R[0]*R[2]/(np.dot(R,R)**0.5)**5
                    E += 0.5*self.fitted_moments[j+len(self.molecule_coords)*7]*R[1]*R[1]/(np.dot(R,R)**0.5)**5
                    E += 0.5*self.fitted_moments[j+len(self.molecule_coords)*8]*R[1]*R[2]/(np.dot(R,R)**0.5)**5
                    E += 0.5*self.fitted_moments[j+len(self.molecule_coords)*9]*R[2]*R[2]/(np.dot(R,R)**0.5)**5
            RMSD += (self.ESP_values[i,0] - E)**2
        RMSD = (RMSD/len(self.surface_points))**0.5
        return RMSD