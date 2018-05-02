import numpy as np
from numba import jit, float64

@jit(float64[:,:,:](float64[:,:],float64[:,:],float64[:,:]))
def Vector_q_A(molecule_coords, point_coords, ESP):
    Vector_out = np.zeros((len(molecule_coords),len(point_coords)))
    for j in range(len(molecule_coords)):
        for i in range(len(point_coords)):
            R = molecule_coords[j,:] - point_coords[i,:]
            Vector_out[j,i] = 1.0/np.dot(R,R)**0.5
    return Vector_out
    

@jit(float64[:,:,:](float64[:],float64[:],float64[:]))
def Vector_mu_A(molecule_coords, point_coords, ESP):
    Vector_out = np.zeros((3,len(molecule_coords),len(point_coords)))
    for j in range(len(molecule_coords)):
        for i in range(len(point_coords)):
            R = molecule_coords[j,:] - point_coords[i,:]
            Vector_out[0,j,i] += R[0]/(np.dot(R,R)**0.5)**3
            Vector_out[1,j,i] += R[1]/(np.dot(R,R)**0.5)**3
            Vector_out[2,j,i] += R[2]/(np.dot(R,R)**0.5)**3
    return Vector_out
    
    
@jit(float64[:,:,:](float64[:],float64[:],float64[:]))
def Vector_theta_A(molecule_coords, point_coords, ESP):
    Vector_out = np.zeros((6,len(molecule_coords),len(point_coords)))
    for j in range(len(molecule_coords)):
        for i in range(len(point_coords)):
            R = molecule_coords[j,:] - point_coords[i,:]
            Vector_out[0,j,i] += 0.5*R[0]*R[0]/(np.dot(R,R)**0.5)**5
            Vector_out[1,j,i] += 0.5*R[0]*R[1]/(np.dot(R,R)**0.5)**5
            Vector_out[2,j,i] += 0.5*R[0]*R[2]/(np.dot(R,R)**0.5)**5
            Vector_out[3,j,i] += 0.5*R[1]*R[1]/(np.dot(R,R)**0.5)**5
            Vector_out[4,j,i] += 0.5*R[1]*R[2]/(np.dot(R,R)**0.5)**5
            Vector_out[5,j,i] += 0.5*R[2]*R[2]/(np.dot(R,R)**0.5)**5
    return Vector_out
    

@jit(float64[:](float64[:,:],float64[:,:],float64[:,:]))
def Vector_q_B(molecule_coords, point_coords, ESP):
    Vector_out = np.zeros(len(molecule_coords))
    for j in range(len(molecule_coords)):
        for i in range(len(point_coords)):
            R = molecule_coords[j,:] - point_coords[i,:]
            Vector_out[j] += ESP[i,0]/np.dot(R,R)**0.5
    return Vector_out
    

@jit(float64[:,:](float64[:],float64[:],float64[:]))
def Vector_mu_B(molecule_coords, point_coords, ESP):
    Vector_out = np.zeros((3,len(molecule_coords)))
    for j in range(len(molecule_coords)):
        for i in range(len(point_coords)):
            R = molecule_coords[j,:] - point_coords[i,:]
            Vector_out[0,j] += ESP[i,0]*R[0]/(np.dot(R,R)**0.5)**3
            Vector_out[1,j] += ESP[i,0]*R[1]/(np.dot(R,R)**0.5)**3
            Vector_out[2,j] += ESP[i,0]*R[2]/(np.dot(R,R)**0.5)**3
    return Vector_out
    
    
@jit(float64[:,:](float64[:],float64[:],float64[:]))
def Vector_theta_B(molecule_coords, point_coords, ESP):
    Vector_out = np.zeros((6,len(molecule_coords)))
    for j in range(len(molecule_coords)):
        for i in range(len(point_coords)):
            R = molecule_coords[j,:] - point_coords[i,:]
            Vector_out[0,j] += 0.5*ESP[i,0]*R[0]*R[0]/(np.dot(R,R)**0.5)**5
            Vector_out[1,j] += 0.5*ESP[i,0]*R[0]*R[1]/(np.dot(R,R)**0.5)**5
            Vector_out[2,j] += 0.5*ESP[i,0]*R[0]*R[2]/(np.dot(R,R)**0.5)**5
            Vector_out[3,j] += 0.5*ESP[i,0]*R[1]*R[1]/(np.dot(R,R)**0.5)**5
            Vector_out[4,j] += 0.5*ESP[i,0]*R[1]*R[2]/(np.dot(R,R)**0.5)**5
            Vector_out[5,j] += 0.5*ESP[i,0]*R[2]*R[2]/(np.dot(R,R)**0.5)**5
    return Vector_out