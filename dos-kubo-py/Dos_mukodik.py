#!/usr/shared_apps/packages/python-2.7.3-build3/bin/python2.7

import sys,time
sys.path.append("/home/jegesm/workspace/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
import petsc4py
import numpy as np
from numpy import ix_
#from scipy.special import jn

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

 USAGE:   python Dos.py rndseed chebdeg outfname hamfile 

"""

if sys.argv[1].find("h")>-1:
 help()
 exit()

rndseed=int(sys.argv[1])
chebdeg=int(sys.argv[2])
outfname=sys.argv[3]
hamfile=sys.argv[4]
one=1.0

#// Setting output file
#OUTviewer = PETSc.Viewer().createASCII(outfname, 'w')
OUT=open(outfname,'w')
#//---------------------------------------------------
#// Importing the Hamiltonian, and the two operators
#// used in the correlation function. 
#// THESE OBJECTS NEED TO BE LOADED !!!!!!
#//---------------------------------------------------

viewer = PETSc.Viewer().createBinary(hamfile, 'r')
H = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
H = PETSc.Mat().load(viewer)
viewer.destroy()
dim = H.getSize()
H.setFromOptions()

#// H is the Hamiltonian 
#//---------------------------------------------------
#//---------------------------------------------------
#// Initiate random vector
#//---------------------------------------------------
psir = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
psir.setSizes(dim)
psir.setFromOptions()
rctx = PETSc.Random().create(comm=PETSc.COMM_WORLD); 
rctx.setFromOptions()
rctx.setInterval([-1.0-1.0j,1.0+1.0j])
rctx.setSeed(rndseed)
psir.setRandom(rctx)
rctx.destroy()
#viewer = PETSc.Viewer().createASCII('ff')
#viewer(psir)
psii=psir.copy()
#psii.view(viewer)

psir.axpy(1.0j,psii)

#psir.view(viewer)
psir.norm(PETSc.NormType.NORM_2)
#psir.view(viewer)
psir.normalize()
#psir.view(viewer)

psi0=psir.copy()
psir.destroy()
psii.destroy()

#// Random vector initialized and is in psi0
#//---------------------------------------------------
#O2psi0 = psi0.copy()
#OP2.mult(psi0,O2psi0);
#// O2|psi0> is also stored
#//---------------------------------------------------

#//Begining of the Chebyshev recursion
psim = psi0.copy()
psimm1 = psi0.copy()
psimp1 = psi0.copy()

#O2psim1 = O2psi0.copy()
#O2psim2 = O2psi0.copy()
#O2psimp1 = O2psi0.copy()

psitmp = psi0.copy()

T1=PETSc.Log().getTime()
  
#//Doing the recursion
for nn in range(chebdeg):

     
  if (nn == 0) :
    #|psi_0>=I|psi>
    psimp1 = psi0.copy()
    psim = psi0.copy()
    
  elif (nn == 1):
    #|psi_1>=H|psi>    
    psimm1 = psim.copy()
    H.mult(psi0,psim)
     
  else:
    #|psi_m+1>=2H|psi_m> - |psi_m-1>    
    psimm1.scale(-one)
    H.mult(psim,psimp1)
    psimp1.aypx(2.0*one,psimm1)
    psimm1 = psim.copy()
    psim = psimp1.copy()
    
    
  ChebMomentum = psi0.dot(psim)
  
  if nn%200==0:
   print nn," "
   vnam="vect_%d"%(nn)
   viewer = PETSc.Viewer().createASCII(vnam)
   viewer(psim)    
   viewer.destroy()
#    TT=PETSc.Object("%f  %f\n" %(ChebKern.real,ChebKern.imag))
#    TT.setFromOptions()
#    OUTviewer("%f  %f\n" %(ChebKern.real,ChebKern.imag))
#    OUTviewer(TT)
  OUT.write("%d %f %f\n"%(nn,ChebMomentum.real,ChebMomentum.imag))
#  OUT.write("%f %f\n"%(ChebKern.real,ChebKern.imag))




OUT.close() 
print "Chebyshev expansion done"
psi0.destroy()
psim.destroy()
psimm1.destroy()
psimp1.destroy()

psitmp.destroy()
H.destroy()

T2=PETSc.Log().getTime()
print T2-T1

