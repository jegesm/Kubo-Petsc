#!/usr/shared_apps/packages/python-2.7.3-build3/bin/python2.7

import sys,time
sys.path.append("/home/jegesm/Install/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
sys.path.append("/home/jegesm/Munka/py-kubo/")
import Utils_petsc
import petsc4py
import numpy as np
from numpy import ix_

petsc4py.init(sys.argv)
from Utils_petsc import *
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
parser.add_argument('--vx',help='Velocity operator x')
parser.add_argument('--fhcoeffs',help='File that stores the f(H) expansion')
parser.add_argument('--nrss',default=1, type=int,help='Numer of random starting vectors')
parser.add_argument('--time',type=int,default=4, help='Time to evolve the state')
parser.add_argument('--dt',type=int,default=1, help='time interval between timesteps')
parser.add_argument('--eps',type=float, default=False, help='Energy')
args = parser.parse_args()

one=1.0
#print_mat(Velx_gr(4,1,3.1,t=-5.55))
print_mat(Vely_gr(4,1,3.1,t=-5.55))
#H=Velx_gr(4,1,3.1,t=-5.55)
#print H
