# Generating a square graphene sheet hamiltonian of dim*dim carbon atoms
# and the jx operator
# and chebyshev coeffs

import sys,time
sys.path.append("/home/jegesm/Install/petsc4py-3.5/build/lib.linux-x86_64-2.7/")
import petsc4py
import numpy as np
from numpy import ix_
from scipy.linalg import norm as Mnorm

petsc4py.init(sys.argv)
from Utils_petsc import *
from Extra import *
from petsc4py import PETSc

import argparse

parser = argparse.ArgumentParser(description='Generate Hamiltonians. .')
parser.add_argument('--type', help='Lattice type: gr, cu')
parser.add_argument('--xdim',type=int,default=1, help='dimension in x direction')
parser.add_argument('--ydim',type=int,default=1, help='dimension in y direction')
parser.add_argument('--zdim',type=int,default=1, help='dimension in z direction')
parser.add_argument('--defdens',type=float,default=0, help='defines the defect density (%)')
parser.add_argument('--deftype',default=0, help='defines the type of the defect')
parser.add_argument('--fhchebdeg',type=int,default=100, help='defines degree of Chebyshev expansion ')
parser.add_argument('-p',type=int,default=1, help='Periodic')
parser.add_argument('--eps',type=float, default=False, help='onsite Energy')
parser.add_argument('--mu',type=float, default=False, help='chemical potential')
parser.add_argument('--kbt',type=float, default=False, help='Temperature')
parser.add_argument('-V',type=int, default=False, help='Velocity operator')
args = parser.parse_args()

#################
rnp=0
tag=""; tagham=""
# CUBIC LATTICE
if args.type=="cu":
    rnp=6.01-(3-min(args.ydim,3))-(2-min(args.xdim,2))-(3-min(args.zdim,3))
    T1=time.time()
    Mat, coords = Make_cubic_ham(args.xdim,args.ydim,args.zdim,args.p,rnp)

    dim=args.xdim*2*args.ydim*args.zdim
    rnp=rnp+abs(args.eps)
    print "rnp, eps:  ", rnp,args.eps
    tmpMat, tmpcoords=Make_cubic_ham(15,15,15,args.p,rnp)

    tag="cu_%d"%(args.xdim)
    tagham="cu_%d"%(args.xdim)
    
    CK=Chebcoeffs(args.kbt,args.mu,args.fhchebdeg)
    CK.conjugate()
    jxop = Make_Jx_op(10,args.p,rnp)
    JX = Create_petsc_mat(jxop,-1.0j/rnp)
    Jxname="jxsq_%d"%(dim)


#################3
# GRAPHENE
if args.type=="gr":
    tag="gr_%d"%(args.xdim)
    tagham="gr_%d"%(args.xdim)
    
    rnp=3.01
    dim=args.xdim
    
    CK=Chebcoeffs(args.kbt,args.mu,args.fhchebdeg)
    Mat, coords = Make_graphene_ham_coo(dim,args.p,rnp,0.0, -1)    
    print "<<  Matrix generated"
    if args.V:
        Vx=Velx_gr(dim,1,0.0)
        Velx=Create_petsc_mat(Vx, 3.01)
        velxname="velxgr_%d"%(dim)
        viewer = PETSc.Viewer().createBinary(velxname, 'w')
        viewer(Velx)
#        print_mat(Vx.todense())
    
    tmpMat, tmpcoords=Make_graphene_ham_coo(50,args.p,rnp)
    
    Jx = Make_Jx_op(dim,args.p,rnp-1)
# Jx = Make_Jx_id_op(G_mat.size,p,1.01)
    Jxname="jxgr_%d"%(dim)
    
    t= -2.
    eps=-t/16.
    
    JX = Create_petsc_mat(Jx,-1.0j/rnp)
    #print Mat.todense()

    

    

if args.deftype=="vac":
        Mat = Vacancy(Mat,args.defdens/100.0, args.eps, 0)
        tmpMat=Vacancy(tmpMat, args.defdens/100,args.eps, 0)
        print "<<  defects placed into Matrix"
        newdim=Mat.shape[0]
        tagham+="_vac"
if args.deftype=="ad":
        t=0
        Mat, coords = Adatom(Mat, coords, args.defdens/100.0, args.eps, t)
        tmpMat, tmpcoords = Adatom(tmpMat, tmpcoords, args.defdens/100.0, args.eps, t)
        print "<<  Adatoms added"
        tagham+="_ad%d"%(args.eps)
if args.deftype=="dot":
        t=0
        R0=np.array([100, 50, 0])
        R=6
        Mat = Dot(Mat,R0, R, coords, args.eps, t)
        tmpMat = Dot(tmpMat, R0, R, tmpcoords, args.eps, t)
        print "<<  Dot added"
        tagham+="_dot%4.2f"%(args.eps)

Mat = AddPot(Mat, 'x',  coords, -10.0, 10.0)
tagham+="_Vx"
    


Mcoords=Create_petsc_mat(coords, 1.)

#FIND THE NORM    
tmpMat.tocsr()
tmpMat=Create_petsc_mat(tmpMat, 1.)
rnp=tmpMat.norm(3)+0.01
print "tmp rnp: ", rnp
Mat.tocsr()
#print Mat.todense()
G_mat=Create_petsc_mat(Mat,1./rnp)
hamname="ham%s"%(tagham)
coordsname="coord%s"%(tag)

print "System size D = 10^"+ str(np.log10(G_mat.size[0])) 

# save
viewer = PETSc.Viewer().createBinary("fhcoeff_"+str(args.fhchebdeg), 'w')
viewer(CK)
viewer = PETSc.Viewer().createBinary(Jxname, 'w')
viewer(JX)

viewer = PETSc.Viewer().createBinary(hamname, 'w')
viewer(G_mat)
viewer = PETSc.Viewer().createASCII(hamname+"-ascii", 'w')
viewer(G_mat)
viewer = PETSc.Viewer().createASCII("df_"+str(args.fhchebdeg))
viewer(CK)

viewer = PETSc.Viewer().createBinary(coordsname, 'w')
viewer(Mcoords)
viewer = PETSc.Viewer().createASCII(coordsname+"-ascii", 'w')
viewer(Mcoords)

# load
#viewer = PETSc.Viewer().createBinary('ham', 'r')
#B = PETSc.Mat().load(viewer)

# check#assert B.equal(A)
