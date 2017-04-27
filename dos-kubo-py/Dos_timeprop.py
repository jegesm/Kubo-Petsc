#!/usr/shared_apps/packages/python-2.7.3-build3/bin/python2.7

import sys,time
sys.path.append("/home/jegesm/Install/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
import Utils_petsc
import petsc4py
import numpy as np
from numpy import ix_
import datetime

petsc4py.init(sys.argv)
from Utils_petsc import *
from petsc4py import PETSc


def help():
 print """
Calculation of the DoS with Chebyshev Expansion. 
<psi0| T_n |psi0> 

-rndseed random seed for the random starting vector 
-chebdeg defines degree of Chebyshev expansion 
-outfname output file name for the Chebyshev Kernel
-hamfile name of file with the hamiltonian                                 (MUST SUPPLY)
-Numrss number of random states

 USAGE:   python Dos.py rndseed chebdeg outfname hamfile 

"""

if sys.argv[1].find("h")>-1:
 help()
 exit()

Numrss=int(sys.argv[8])

rndseed=int(sys.argv[1])
fhchebdeg=int(sys.argv[2])
chebdeg=int(sys.argv[3])
outfname=sys.argv[4]
hamfile=sys.argv[5]
fhcf=sys.argv[6]
T=int(sys.argv[7])
one=1.0
Stime = np.datetime64(datetime.datetime.now()).item().second

for N in range(Numrss):
    print N
    #// Setting output file
    #OUTviewer = PETSc.Viewer().createASCII(outfname, 'w')
    
    #//---------------------------------------------------
    #// Importing the Hamiltonian, and the two operators
    #// used in the correlation function. 
    #// THESE OBJECTS NEED TO BE LOADED !!!!!!
    #//---------------------------------------------------
    H, dim=LoadPetscMat(hamfile)
    fhcoeffs, dimfhcoeffs=LoadPetscVec(fhcf)
    
    #// Initiate random vector
    psi0 = RandomVec(dim,[-1.0-1.0j,1.0+1.0j], args.rndseed)
    psir2 = RandomVec(dim,[-1.0-1.0j,1.0+1.0j], args.rndseed)
    psi0.axpy(1.0j,psir2)    
    psi0.normalize()    
    
    #//Begining of the Chebyshev recursion
    psim = psi0.copy()
    psimm1 = psi0.copy()
    psimp1 = psi0.copy()
    psitmp = psi0.copy()
    
    #//Doing the recursion
    OUT=open(outfname+"-"+str(N),'w')
    psit,ChebMomentum = Chebyshev_F_rec(H,psi0,fhchebdeg,fhcoeffs)
    tcorr=psit.dot(psi0)

    dt=1
    #//Doing the recursion
    for time in np.arange(0,T,dt):
        OUT.write("%f %f %f\n"%(time,tcorr.real,tcorr.imag))

        psitpdt=Timeprop(H,psit,chebdeg,dt)	#; print "1 ",psi1tpdt.norm()
        tcorr=psitpdt.dot(psi0)	#; print "3 ",psitmp.norm(),tcorr
        psit=psitpdt.copy()
    
    OUT.close() 
    
    print "Chebyshev expansion done"
    
    psi0.destroy();    psim.destroy()
    psimm1.destroy();    psimp1.destroy()
    psitmp.destroy();    H.destroy()

