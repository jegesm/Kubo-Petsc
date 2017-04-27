#!/usr/shared_apps/packages/python-2.7.3-build3/bin/python2.7

import sys,time
sys.path.append("/home/jegesm/Install/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
sys.path.append("/home/jegesm/Install/petsc-3.5.2/new/lib")
sys.path.append("../")
import Utils_petsc
import petsc4py
import numpy as np
from numpy import ix_

petsc4py.init(sys.argv)
from Utils_petsc import *
from Extra import *
from petsc4py import PETSc
import argparse

parser = argparse.ArgumentParser(description='Calculation of the OC with Chebyshev Expansion. .')
parser.add_argument('--mode', help='Mode: oc, dos')
parser.add_argument('--rndseed',type=int,default=-1, help='Seed for random vector generation (default computer generated)')
parser.add_argument('--chebdeg',type=int,default=100, help='defines degree of Chebyshev expansion ')
parser.add_argument('--fhchebdeg',type=int,default=100, help='defines degree of Chebyshev expansion ')
parser.add_argument('--out',default="OUT", help='output file name for the Chebyshev momenta')
parser.add_argument('--ham',help='The hamiltonian of the system')
parser.add_argument('--op1',help='Operator 1')
parser.add_argument('--op2',help='Operator 2')
parser.add_argument('--coords',help='Coords')
parser.add_argument('--vx',help='Velocity operator x')
parser.add_argument('--fhcoeffs',help='File that stores the f(H) expansion')
parser.add_argument('--nrss',default=1, type=int,help='Numer of random starting vectors')
parser.add_argument('--time',type=int,default=4, help='Time to evolve the state')
parser.add_argument('--dt',type=int,default=1, help='time interval between timesteps')
parser.add_argument('--eps',type=float, default=False, help='Energy')
args = parser.parse_args()

#psi0 = RandomVec(4,args.rndseed)
##BF=psi0.array
#print dir(F)
#exit()
one=1.0
H, dim=LoadPetscMat(args.ham)
if args.mode=="prop":
    sigma= 10
    coords, dim = LoadPetscMat(args.coords)
    print dim
    #print dir(coords)
    coo=coords.getValues(range(dim[0]), range(dim[1]))
    #coo=coords
    R0=0
    psi0 = GaussianPacketVec(dim[0], sigma, R0, coo)
    psit=psi0.copy()
    lab=1
    for time in np.arange(0,args.time,args.dt):
        psitmp=Timeprop(H, psit, args.chebdeg, args.dt)
        psit=psitmp.copy()
        viewer = PETSc.Viewer().createASCII("vec_%05d"%(lab))
        viewer(psit)
        lab+=1
        
        
if args.mode=="dosm":
    psi0 = RandomVec(dim,args.rndseed)
    psir2 = RandomVec(dim, args.rndseed)
    psi0.axpy(1.0j,psir2)    
    psi0.normalize()  
    Mayoudos(H, dim, psi0, args.nrss, args)     
if args.mode=="qeig":
    psi0 = RandomVec(dim,args.rndseed)
    MayouQeig(H, dim, psi0, args)     
    
if args.mode=="mdc":
    psi0 = RandomVec(dim,args.rndseed)
    MayouDC(H, dim, psi0, args)     

if args.mode=="qch":
    Quasicheck(H, psi0, args)
    
exit()
for N in range(args.nrss):
#for N in range(0):
    rndseed=args.rndseed
    print N 
    #// Initiate random vector    
    psi0,  rndseed= RandomVec(dim,rndseed)
    psir2, rndseed = RandomVec(dim,rndseed)
    psi0.axpy(1.0j,psir2)    
    psi0.normalize()  
    
    if args.mode=="oc":
        OC_timeprop(H, psi0, N, args)
    if args.mode=="dos":
        Dos_timeprop(H, psi0, N, args)
    if args.mode=="qe":
        QE_timeprop(H, psi0, N, args)
    if args.mode=="simados":
        Dos(H, psi0, args)    
    if args.mode=="simaoc":
        OC(H, psi0, args)    
    if args.mode=="dc":
        DC_timeprop(H, psi0, N, args)
    
    psi0.destroy();   

      
H.destroy()

 
