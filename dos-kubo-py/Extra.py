import sys,time
sys.path.append("/home/jegesm/Install/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
sys.path.append("/home/jegesm/Install/petsc-3.5.2/new/lib")
sys.path.append("../")
import Utils_petsc
import petsc4py
from petsc4py import PETSc
import numpy as np

#import h5py


def GaussianPacketVec(dim, sigma, R0, coo):

    psi = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    psi.setSizes(dim)
    psi.setFromOptions()
    
    Kx=np.array([-1.0, 0.0, 0.0])
    R0=np.array([50.0,0.0, 0.0])
    R=coo-R0
    RK=np.dot(R,Kx)
    #G=np.exp(1.0j*(RK))*np.exp(-np.linalg.norm(R,axis=1)**2)
    
    R=coo[:, 0]-R0[0]
    RK=R*Kx[0]
    G=np.exp(1.0j*(RK))*np.exp(-R**2)
    psi.setValues( range(dim), values=G)
    psi.normalize()
    
    return  psi
    
def  saveh5():
    Fh5 = h5py.File(outfile+".h5", "w")
    rset = Fh5.create_dataset("Rho",shape=Rho.shape, data=Rho, dtype='f')
    xset = Fh5.create_dataset("x", shape=X.shape,data=X, dtype='f')
    yset = Fh5.create_dataset("y", shape=Y.shape,data=Y, dtype='f')
    zset = Fh5.create_dataset("z", shape=Z.shape,data=Z, dtype='f')

    Fh5.close()

def AddPot(Mat, dir, coords,  Vmin,  Vmax):
    xmin, ymin, zmin = np.min(coords, axis=0)
    xmax, ymax, zmax = np.max(coords, axis=0)
    if dir=="x":
        dV = (Vmax-Vmin)/(xmax-xmin)
        for i in range(len(coords)):
            Hv = (Vmin+coords[i][0]-xmin)*dV
            Mat[i, i] += Hv
    
    if dir=="y":
        dV = (Vmax-Vmin)/(ymax-ymin)
        for i in range(len(coords)):
            Hv = (Vmin+(coords[i][1]-ymin)*dV)
            Mat[i, i] += Hv
        
    return Mat
