#!/usr/shared_apps/packages/python-2.7.3-build3/bin/python2.7

import sys,time
sys.path.append("/home/jegesm/Install/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
import Utils_petsc
import petsc4py
import numpy as np
from numpy import ix_

petsc4py.init(sys.argv)
from Utils_petsc import *
from petsc4py import PETSc
import argparse

parser = argparse.ArgumentParser(description='Calculation of the OC with Chebyshev Expansion. .')
parser.add_argument('--rndseed',type=int,default=-1, help='Seed for random vector generation (default computer generated)')                                      
parser.add_argument('--chebdeg',type=int,default=100, help='defines degree of Chebyshev expansion ')                                      
parser.add_argument('--fhchebdeg',type=int,default=100, help='defines degree of Chebyshev expansion ')                                      
parser.add_argument('--out',default="OUT", help='output file name for the Chebyshev momenta')                                      
parser.add_argument('--ham',help='The hamiltonian of the system')                                      
parser.add_argument('--op1',help='Operator 1')                                      
parser.add_argument('--op2',help='Operator 2')                                      
parser.add_argument('--fhcoeffs',help='File that stores the f(H) expansion')                                      
parser.add_argument('--nrss',default=1, type=int,help='Numer of random starting vectors')                   
parser.add_argument('--time',type=int,default=4, help='Time to evolve the state')                                      
parser.add_argument('--dt',type=int,default=1, help='time interval between timesteps')                                      
args = parser.parse_args()

one=1.0

for N in range(args.nrss):
    print N
    #// Setting output file
    #OUTviewer = PETSc.Viewer().createASCII(outfname, 'w')
    
    #//---------------------------------------------------
    #// Importing the Hamiltonian, and the two operators
    #// used in the correlation function. 
    #// THESE OBJECTS NEED TO BE LOADED !!!!!!
    #//---------------------------------------------------
    H, dim=LoadPetscMat(args.ham)
    OP1, dim=LoadPetscMat(args.op1)
    OP2, dim=LoadPetscMat(args.op2)
    fhcoeffs, dimfhcoeffs=LoadPetscVec(args.fhcoeffs)
    
    #// Initiate random vector    
    psi0 = RandomVec(dim,args.rndseed)
    psir2 = RandomVec(dim, args.rndseed)
    psi0.axpy(1.0j,psir2)    
    psi0.normalize()  
    
    
    #//Begining of the Chebyshev recursion
    psim = psi0.copy()
    psitmp = psi0.copy()
    
    #//Doing the recursion
    OUT=open(args.out+"-"+str(N),'w')
    OP1.mult(psi0, psitmp)
    psit1,ChebMomentum = Chebyshev_F_rec(H,psitmp,args.fhchebdeg,fhcoeffs)
    psit2,ChebMomentum = Chebyshev_F_rec(H,psi0,args.fhchebdeg,fhcoeffs)
    psit1.axpy(-1, psitmp)
    tcorr=psit2.dot(psit1)

    dt=1
    #//Doing the recursion
    for time in np.arange(0,args.time,args.dt):
        OUT.write("%f %f %f\n"%(time,tcorr.real,tcorr.imag))

        psitpdt1=Timeprop(H,psit1,args.chebdeg,dt)	#; print "1 ",psi1tpdt.norm()
        psitpdt2=Timeprop(H,psit2,args.chebdeg,dt)	#; print "1 ",psi1tpdt.norm()
        OP2.mult(psitpdt1, psitmp)
        tcorr=psitpdt2.dot(psitmp)	#; print "3 ",psitmp.norm(),tcorr
        psit1=psitpdt1.copy()
        psit2=psitpdt2.copy()
    
    OUT.close() 
    
    print "Chebyshev expansion done"
    
    psi0.destroy();    psim.destroy()
    psimm1.destroy();    psimp1.destroy()
    psitmp.destroy();    H.destroy()
    OP1.destroy();    OP2.destroy()
