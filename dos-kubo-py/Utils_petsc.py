# Generating a square graphene sheet hamiltonian of dim*dim carbon atoms
# and the jx operator
# and chebyshev coeffs

#ONE MIGHT NEED TO INVOKE THESE COMMANDS
import sys,time
#sys.path.append("/storage/users/visontai/petsc4py-3.3/build/lib.linux-x86_64-2.7/")
#import petsc4py
import numpy as np
from numpy import ix_, exp, cos, pi
from scipy.linalg import norm as Mnorm
from scipy.special import jn
import time
from progressbar import Percentage, ProgressBar, Bar
import datetime
#petsc4py.init(sys.argv)

from petsc4py import PETSc

#Here is all the stuff that is needed to initialize and finalize a PETSC Matrix...

def Create_petsc_mat(Mat,alpha):
# T1=time.time()
 #GET SIZE OF NUMPY MATRIX
 m,n = Mat.shape[0],Mat.shape[1]

 # INIT PETSC MATRIX
 A = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
 A.setSizes([m,n]);  A.setFromOptions();  A.setUp()
 Istart, Iend = A.getOwnershipRange()

# print "Converting to PETSc Matrix"

 #COPY MATRIX
 try:
    for (r,c),v in Mat.iteritems():
        A[r,c]=v*alpha
 except AttributeError:
    for x in range(Mat.shape[0]):
        for y in range(Mat.shape[1]):
            A[x, y] = Mat[x, y]


 #FINALIZE
 A.assemblyBegin();  A.assemblyEnd()

# print time.time()-T1
 return A


#Here is all the stuff that is needed to initialize and finalize...
def Create_petsc_vec(Vec):
# T1=time.time()
 # GET LENGTH OF LIST/NUMPY.ARRAY
 m = len(Vec)

 # INIT PETSC VECTOR
 A = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
 A.setSizes(m);  A.setFromOptions(); A.setUp()
 Istart, Iend = A.getOwnershipRange()

# print "Converting to PETSc Vector"

 # COPY VECTOR
 for u in range(m):
  A[u]=Vec[u]

 #FINALIZE
 A.assemblyBegin(); A.assemblyEnd()

# print time.time()-T1
 return A


###################################################33
# GRAPHENE
def Make_graphene_ham(dim,p,rnp, eps=0.0, t=-1):
 from scipy import sparse as sp
 from scipy.sparse import dok_matrix

 """ 
 dim dimension
 p periodic boundary cond
 rnp hamiltonian renormalization parameter
 """

 #Start timing
# T1=time.time()
 d=sp.eye(dim,dim)

 d_o = np.eye(dim)
 d_o[1::2,1::2]=0
 d_e = np.eye(dim)
 d_e[::2,::2]=0
 d_e=dok_matrix(d_e)
 d_o=dok_matrix(d_o)
# print "d,d_o,d_e = ",time.time()-T1;  T1=time.time()

 D1 = np.ones(dim-1)
 od=sp.diags(D1,1)
 #print Mat.todense()
 od_p = od.T
 od = dok_matrix(od)
 od_p = dok_matrix(od_p)
# print "od = ", time.time()-T1 ; T1=time.time()

 D2 = np.array([ i%2 for i in range(dim-1) ])
 odo = sp.diags(D2,1)
 odo = dok_matrix(odo)
 odo[0,-1] = 1
 odo_p = odo.T

 D2 = np.array([ i%2 for i in range(1,dim) ])
 ode = sp.diags(D2,1) 
 ode = dok_matrix(ode)
 ode_p = ode.T
# print "odo = ", time.time()-T1;  T1=time.time()

 M1 = dok_matrix((dim*dim,dim*dim))
 M2 = dok_matrix((dim*dim,dim*dim))
 for i in range(dim):
  M1[i,i+dim*dim-dim] = 1
  M2[i+dim*dim-dim,i] = 1
  # print "M = ", time.time()-T1;  T1=time.time()

 Gr_mat = t*(-(sp.kron(d_e,ode+ode_p)+sp.kron(d_o,odo+odo_p)+\
  #sp.kron(od+od_p,d)) - p*M1 - p*M2) + sp.eye(dim*dim)*eps
  sp.kron(od+od_p,d)) ) + sp.eye(dim*dim)*eps
 
 Gr_mat=dok_matrix(Gr_mat)
 
 return Gr_mat
    

def Make_graphene_ham_coo(dim,p,rnp, eps=0.0, t=-1):
    from scipy import sparse as sp
    from scipy.sparse import dok_matrix
    
    """ 
    dim dimension
    p periodic boundary cond
    rnp hamiltonian renormalization parameter
    """
    
    #Start timing
    # T1=time.time()
    d=sp.eye(dim,dim)
    
    d_o = np.eye(dim)*4.55
    d_o[1::2,1::2]=0
    d_e = np.eye(dim)*4.55
    d_e[::2,::2]=0
    d_e=dok_matrix(d_e)
    d_o=dok_matrix(d_o)
    # print "d,d_o,d_e = ",time.time()-T1;  T1=time.time()
    
    D1 = np.ones(dim-1)
    od=sp.diags(D1,1)
    #print Mat.todense()
    od_p = od.T
    od = dok_matrix(od)
    od_p = dok_matrix(od_p)
    # print "od = ", time.time()-T1 ; T1=time.time()
    
    D2 = np.array([ i%2 for i in range(dim-1) ])
    odo = sp.diags(D2,1)
    odo = dok_matrix(odo)
    odo[0,-1] = 2.33
    odo_p = odo.T
    
    D2 = np.array([ i%2 for i in range(1,dim) ])
    ode = sp.diags(D2,1) 
    ode = dok_matrix(ode)
    ode_p = ode.T
    # print "odo = ", time.time()-T1;  T1=time.time()
    
    M1 = dok_matrix((dim*dim,dim*dim))
    M2 = dok_matrix((dim*dim,dim*dim))
    for i in range(dim):
        M1[i,i+dim*dim-dim] = 1
        M2[i+dim*dim-dim,i] = 1
    # print "M = ", time.time()-T1;  T1=time.time()
    
    
    #Gr_mat = t*(-(sp.kron(d_e,ode+ode_p)+sp.kron(d_o,odo+odo_p)+\
    Gr_mat = t*(-(sp.kron(d_e,ode+ode_p)+sp.kron(d_o,odo+odo_p)+\
    sp.kron(od+od_p,d)) - p*M1 - p*M2) + sp.eye(dim*dim)*eps
    #sp.kron(od+od_p,d)) + sp.eye(dim*dim)*eps
    #print_mat(Gr_mat.todense())
    #print Gr_mat
    
    Gr_mat=dok_matrix(Gr_mat)
    
    a=1.43
    sqa=np.sqrt(3)/2*a
    coords=np.zeros((dim*dim, 3))
    z=0.0
    for i in range(dim):
        x=0
        y=i*a+i/2*a
        coords[i]=[x, y, z]
        y=(i+1)*a+i/2*a
        coords[i+dim]=[x+sqa, y+a/2, z]
        for j in range(1, dim/2):
            coords[i+j*dim*2]=coords[i]+[j*2*sqa, 0, 0]
            coords[i+dim+j*dim*2]=coords[i+dim]+[j*2*sqa, 0, 0]
    
    return Gr_mat, coords

###################################################################################
# CUBIC LATTICE LATTICE
def Make_cubic_ham(xdim, ydim, zdim,p,rnp):
    from scipy import sparse as sp
    from scipy.sparse import dok_matrix

    """ 
    dim=xdim*ydim*zdim  x,y,z dimensions
    p periodic boundary cond
    rnp hamiltonian renormalization parameter
    """
    
    dim=xdim*ydim*zdim
    eps=0.0
    hop=1.0
    #Start timing
    T1=time.time()
    #columns: its size will be 2*xdim
    D=np.empty(xdim)
    D.fill(1)
    D1=np.empty(xdim-1)
    D1.fill(1)
    #around a site
    hu=np.array([[eps,hop],[hop,eps]])
    hu=dok_matrix(hu)
    h1x=np.array([[0,0],[hop,0]])
    h1x=dok_matrix(h1x)
    h1xp=h1x.T
    if xdim>1:
        HX=sp.kron(sp.eye(xdim),hu)+sp.kron(sp.diags(D1,1),h1x)+sp.kron(sp.diags(D1,-1),h1xp)
    else:
        HX=hu
    #pbc
    if (p==1):
        HX=HX + sp.kron(sp.diags(1,xdim-1),h1xp) + sp.kron(sp.diags(1,-xdim+1),h1x) 
    # print HX.todense()
    
    #between columns, in Y direction
    E=sp.eye(xdim)
    h1uy=np.array([[hop,0],[0,hop]])
    h1uy=dok_matrix(h1uy)
    HY=sp.kron(E,h1uy)
    # print "HY ", time.time()-T1 ; T1=time.time()
    # if p==1:
    #  HC=HC+sp.kron(sp.diags(1,xdim-1),h1p) + sp.kron(sp.diags(1,-xdim+1),h1) 
    # print "HY"
    # print HY.todense()

    #2D matrix
    F=sp.eye(ydim)
    F1=np.empty(ydim-1)
    F1.fill(1)
    if ydim>1:
        HXY=sp.kron(F,HX)+sp.kron(sp.diags(F1,1),HY)+sp.kron(sp.diags(F1,-1),HY.T)
    else:
        HXY=HX
    # print "HXY"
    # print HXY.todense()
    # print "HXY ", time.time()-T1 ; T1=time.time()
    
    #between planes into z direction
    E=sp.eye(ydim*xdim)
    h1uz=np.array([[hop,0],[0,hop]])
    HZ=sp.kron(E,h1uz)
    # if[==1:
    #  HZ=HZ+
    # print HZ.todense()
    
    F=sp.eye(zdim)
    F1=np.empty(zdim-1)
    F1.fill(1)
    if zdim==1:
        Mat=sp.kron(F,HXY)
    else:
        Mat=sp.kron(F,HXY) + sp.kron(sp.diags(F1,1),HZ)+sp.kron(sp.diags(F1,-1),HZ)
    # print Mat.todense()
    #
    # print "Mat ", time.time()-T1 ; T1=time.time()
    # print "rnp= ",rnp
    # rnp=Mnorm(Mat.todense(),1)+0.01
    print "rnp= ",rnp
    # print mat
    Mat=dok_matrix(Mat)
    
    # print "End ",time.time()-T1
    # return Create_petsc_mat(Mat,1./rnp)
    coords=np.zeros((2*xdim*ydim*zdim, 3))
    for x in range(xdim*2):
        for y in range(ydim):
            for z in range(zdim):
                coords[z+y*zdim+x*ydim]=np.array([x, y, z])
                
    return Mat,  coords

#def Fhbz():
    #H0=np.array([])
    
def Make_cubic_ham_ch(xdim, ydim, zdim,p,H0):
    from scipy import sparse as sp
    from scipy.sparse import dok_matrix

    """ 
    dim=xdim*ydim*zdim  x,y,z dimensions
    p periodic boundary cond
    rnp hamiltonian renormalization parameter
    """
    
    dim=xdim*ydim*zdim
    #Start timing
    T1=time.time()
    #columns: its size will be 2*xdim
    D=np.empty(xdim)
    D.fill(1)
    D1=np.empty(xdim-1)
    D1.fill(1)
    #around a site
 #   hu=np.array([[eps,hop],[hop,eps]])
    H0=dok_matrix(H0)
    H1x=dok_matrix(H1x)
    h1xp=h1x.T
    if xdim>1:
        HX=sp.kron(sp.eye(xdim),hu)+sp.kron(sp.diags(D1,1),h1x)+sp.kron(sp.diags(D1,-1),h1xp)
    else:
        HX=hu
    #pbc
    if (p==1):
        HX=HX + sp.kron(sp.diags(1,xdim-1),h1xp) + sp.kron(sp.diags(1,-xdim+1),h1x) 
    # print HX.todense()
    
    #between columns, in Y direction
    E=sp.eye(xdim)
    h1uy=np.array([[hop,0],[0,hop]])
    h1uy=dok_matrix(h1uy)
    HY=sp.kron(E,h1uy)
    # print "HY ", time.time()-T1 ; T1=time.time()
    # if p==1:
    #  HC=HC+sp.kron(sp.diags(1,xdim-1),h1p) + sp.kron(sp.diags(1,-xdim+1),h1) 
    # print "HY"
    # print HY.todense()

    #2D matrix
    F=sp.eye(ydim)
    F1=np.empty(ydim-1)
    F1.fill(1)
    if ydim>1:
        HXY=sp.kron(F,HX)+sp.kron(sp.diags(F1,1),HY)+sp.kron(sp.diags(F1,-1),HY.T)
    else:
        HXY=HX
    # print "HXY"
    # print HXY.todense()
    # print "HXY ", time.time()-T1 ; T1=time.time()
    
    #between planes into z direction
    E=sp.eye(ydim*xdim)
    h1uz=np.array([[hop,0],[0,hop]])
    HZ=sp.kron(E,h1uz)
    # if[==1:
    #  HZ=HZ+
    # print HZ.todense()
    
    F=sp.eye(zdim)
    F1=np.empty(zdim-1)
    F1.fill(1)
    Mat=sp.kron(F,HXY) + sp.kron(sp.diags(F1,1),HZ)+sp.kron(sp.diags(F1,-1),HZ)
    # print Mat.todense()
    #
    # print "Mat ", time.time()-T1 ; T1=time.time()
    # print "rnp= ",rnp
    # rnp=Mnorm(Mat.todense(),1)+0.01
    print "rnp= ",rnp
    # print mat
    Mat=dok_matrix(Mat)
    
    # print "End ",time.time()-T1
    # return Create_petsc_mat(Mat,1./rnp)
    return Mat




###################################################################################
# HOLSTEIN MODEL
# model for spinless fermions
# http://www.sciencedirect.com/science/article/pii/0003491659900028

def Make_Holstein_ham(dim, om, g, n,p,rnp):
    
 from scipy import sparse as sp
 from scipy.sparse import dok_matrix

 """ 
 dim=xdim*ydim*zdim  x,y,z dimensions
 p periodic boundary cond
 rnp hamiltonian renormalization parameter
 """

 dim=xdim*ydim*zdim

 #Start timing
 T1=time.time()
#columns: its size will be 2*xdim
 D=np.empty(xdim)
 D.fill(1)
 D1=np.empty(xdim-1)
 D1.fill(1)
 #around a site
 
 F=sp.eye(zdim)
 F1=np.empty(zdim-1)
 F1.fill(1)
 Mat=sp.kron(F,HXY) + sp.kron(sp.diags(F1,1),HZ)+sp.kron(sp.diags(F1,-1),HZ)

 Mat=dok_matrix(Mat)
 
# print "End ",time.time()-T1
# return Create_petsc_mat(Mat,1./rnp)
 return Mat


def Vacancy(Mat,dens, eps, t):
    import scipy.sparse as sp
    
    dim=Mat.get_shape()[0]
    listF=np.random.uniform(0, dim,dens*dim)
    listI=set([int(F) for F in listF])
    listI=list(listI)
    ind=np.zeros(dim)
    ind[listI]=1
    
    indMat=sp.csr_matrix(sp.diags(ind, 0))
    indMat[listI, listI]=1
    Mat.tocsr()
    Md=Mat.dot(indMat)
    Md=Md+Md.T
    nonz=np.nonzero(Md)
    nonzz=zip(nonz[0], nonz[1])
    
    ujMat = Mat+indMat*eps
    for i, j in nonzz:
        ujMat[i, j]=t
    
    return sp.dok_matrix(ujMat)

def Dot(Mat,R0, R, coords, eps, t):
    import scipy.sparse as sp
    dim=Mat.get_shape()[0]
    listi=np.zeros((dim))
    lab=0
    coordsR=coords-R0
    for C in coordsR:
        dist=np.linalg.norm(C)
        if dist<R:
           listi[lab]=1
        if R<dist and dist<R*2.3:
           listi[lab]=np.exp(-(dist/R)**2+1)
        lab+=1
    
    ind=np.zeros(dim)
    ind[np.nonzero(listi)]=1
    indMat=sp.csr_matrix(sp.diags(ind, 0))
    indMat[np.nonzero(listi), np.nonzero(listi)]=1
    Mat.tocsr()
    Md=Mat.dot(indMat)
    Md=Md+Md.T
    nonz=np.nonzero(Md)
    nonzz=zip(nonz[0], nonz[1])
    
    
    ujMat = Mat + indMat
    #for i in np.nonzero(listi)[0]:
    for i, j in nonzz:
        ujMat[i, i]=eps*(listi[i])
        ujMat[j, j]=eps*(listi[j])
        #print i, j, ujMat[i, i]
        #print i, listi[i], ujMat[i, i]
    #for i, j in nonzz:
     #   ujMat[i, j]=t
    
    return sp.dok_matrix(ujMat)


def Anderson(Mat, W):
    #ANDERSON MODEL
    dim=Mat.get_shape()[0]
    Eps=np.random.uniform(-W/2,W/2,size=dim)
    #print Eps, W
    for R in range(dim):
        Mat[R,R]=Eps[R]
    
    return Mat
    
def Adatom(Mat, coords, dens, eps, t):
    dim=Mat.get_shape()[0]
    listF=np.random.randint(0, dim,int(dens*dim))
    listI=set([int(F) for F in listF])
    listI=list(listI)
    Nad=len(listI)
    
#    for n in range(Nad):
#        Mat[listI[n], listI[n]]=eps
    
    Mat.resize((dim+Nad, dim+Nad))
    #coords.resize((dim+Nad, 3))
    for n in range(Nad):
        Mat[dim+n, dim+n] = eps
        Mat[listI[n], dim+n] = t
        Mat[dim+n, listI[n]] = t
        #coords[dim+n] = coords[listl[n]]+np.array([0.0, 0.0, 1.0])
        coords = np.vstack((coords, coords[listI[n]]+np.array([0.0, 0.0, 1.0])))
        
#        print listI[n]
#    
        
    #REORDER MATRIX
#    indices = range(dim)
#    for n in range(Nad):
#        indices.insert(listI[n], dim+n)
#    Mat.tocsr()
#    Mat.data[Mat.indptr[0] :MatX.indptr[16]]
    #ujMat=Mat[indices, indices]
    
    return Mat, coords

##########################################
# JX OP
def Make_Jx_op(dim,p,rnp):
 from scipy import sparse as sp
 from scipy.sparse import dok_matrix

 """ 
 dim dimension
 p periodic boundary cond
 rnp hamiltonian renormalization parameter
 """

 #Start timing
# T1=time.time()
 d=sp.eye(dim,dim)

 d_o = np.eye(dim)
 d_o[1::2,1::2]=0
 d_e = np.eye(dim)
 d_e[::2,::2]=0
 d_e=dok_matrix(d_e)
 d_o=dok_matrix(d_o)
# print "d,d_o,d_e = ",time.time()-T1;  T1=time.time()

 D=np.ones(dim-1)
 od=sp.diags(D,1)
 od_p = od.T
 od = dok_matrix(od)
 od_p = dok_matrix(od_p)
# print "od = ", time.time()-T1;  T1=time.time()

 M1 = dok_matrix((dim*dim,dim*dim))
 M2 = dok_matrix((dim*dim,dim*dim))
 for i in range(dim):
  M1[i,i+dim*dim-dim] = 1
  M2[i+dim*dim-dim,i] = 1

# print "M = ", time.time()-T1; T1=time.time()

 Jx=-(sp.kron(od-od_p,d)) +p*M1 - p*M2
 Jx=dok_matrix(Jx)
# print time.time()-T1
# return Create_petsc_mat(Jx,-1.0j/rnp)
# mat=Jx.toarray()
# rnp=Mnorm(mat,1)+0.01
 #print_mat(Jx.todense())
# print "calc rnp= ",rnp
 return Jx


def Make_Jx_chain_op(dim,p,rnp):
 from scipy import sparse as sp
 from scipy.sparse import dok_matrix

 """ 
 dim dimension
 p periodic boundary cond
 rnp hamiltonian renormalization parameter
 """

 #Start timing
# T1=time.time()
 d=sp.eye(dim,dim)

 Jx=d
 Jx=dok_matrix(Jx)
# print time.time()-T1
# return Create_petsc_mat(Jx,-1.0j/rnp)
# mat=Jx.toarray()
# rnp=Mnorm(mat,1)+0.01
# print "rnp= ",rnp
 return Create_petsc_mat(Jx,-1.0/rnp)
 
def Make_Jx_id_op(dim,p,rnp):
 from scipy import sparse as sp
 from scipy.sparse import dok_matrix

 d=sp.eye(dim[0])
 Jx=dok_matrix(d)
 return Create_petsc_mat(Jx,-1.0/rnp)


def X_gr(dim, t):
    un=np.array([[0.0, 0.0], [0.0, 2.0]])
    Xun=un.copy()
    for i in range(int(dim/2)):
        Xun = np.vstack((Xun, un+[0.0,(i+1) * 3.0]))
    
    #Vx = np.vstack((Vx, un[dim%4]))
    
    Yun = Xun.copy()
    for i in range(dim):
        Yun = np.vstack((Yun, Xun+[np.sqrt(3)/2*(i+1), 0 ]))
    
    return Yun

def Velx_gr(dim,p,rnp, eps=0.0, t=-1):
    from scipy import sparse as sp
    from scipy.sparse import dok_matrix
    
    """ 
    dim dimension
    p periodic boundary cond
    rnp hamiltonian renormalization parameter
    """

    #Start timing
    # T1=time.time()
    d=sp.eye(dim,dim)


    # print "d,d_o,d_e = ",time.time()-T1;  T1=time.time()

    D1 = np.ones(dim-1)
    od=sp.diags(D1,1)
    #print Mat.todense()
    od_p = od.T
    od = dok_matrix(od)
    od_p = dok_matrix(od_p)

    M1 = dok_matrix((dim*dim,dim*dim))
    M2 = dok_matrix((dim*dim,dim*dim))
    for i in range(dim):
        M1[i,i+dim*dim-dim] = 1
        M2[i+dim*dim-dim,i] = 1
    # print "M = ", time.time()-T1;  T1=time.time()

    Vx_mat = t*(-(sp.kron(od+od_p,d)) ) + sp.eye(dim*dim)*eps
    
    Vx_mat=dok_matrix(Vx_mat)
    
    return Vx_mat
    
def Vely_gr(dim,p,rnp, eps=0.0, t=-1):
    from scipy import sparse as sp
    from scipy.sparse import dok_matrix
    
    """ 
    dim dimension
    p periodic boundary cond
    rnp hamiltonian renormalization parameter
    """

    #Start timing
    # T1=time.time()
    d=sp.eye(dim,dim)

    d_o = np.eye(dim)
    d_o[1::2,1::2]=0
    d_e = np.eye(dim)
    d_e[::2,::2]=0
    d_e=dok_matrix(d_e)
    d_o=dok_matrix(d_o)
    # print "d,d_o,d_e = ",time.time()-T1;  T1=time.time()

    D2 = np.array([ i%2 for i in range(dim-1) ])
    odo = sp.diags(D2,1)
    odo = dok_matrix(odo)
    odo[0,-1] = 1
    odo_p = odo.T

    D2 = np.array([ i%2 for i in range(1,dim) ])
    ode = sp.diags(D2,1) 
    ode = dok_matrix(ode)
    ode_p = ode.T
    # print "odo = ", time.time()-T1;  T1=time.time()

    Vy_mat = t*(-sp.kron(d_e,ode+ode_p)+sp.kron(d_o,odo+odo_p)) + sp.eye(dim*dim)*eps
    
    Vy_mat=dok_matrix(Vy_mat)
    
    return Vy_mat
    
def Chebcoeffs(kbt,mu,n):

 """ 
 kbt Boltzamnn*Temperature
 mu chemical potential
 n  is a fucking big nuber. n/2 is roughly the number of useful coefficients

 calculating Chebysev coefficients for the Fermi-Dirac distribution
 """

 ck = [(1./(exp((cos(2*pi/n*x)-mu)/kbt)+1)) for x in range(n)]
 fftck=(2./n*np.fft.fft(ck)).real

 return Create_petsc_vec(fftck)


# Modified Chebyshev recursion for exponential function
def Chebyshev_F_rec(Mat,psi0,fhchebdeg,coeffs,  im=1.0):    
    Momentum=[]
 
# |psi0>=I|psi>
    psim=psi0.duplicate()  
    psimp1=psi0.duplicate()
    fhpsi=psi0.duplicate()
    psimm1=psi0.copy()

# |psi1>=H|psi>     
    Mat.mult(psi0,psim)
# print "psim ",psim.norm()

    fhpsi.axpy(0.5*coeffs[0],psimm1)
# print "fhpsi ",fhpsi.norm()
    Momentum.append(psi0.dot(psimm1))
 
    fhpsi.axpy(1.0*im*coeffs[1],psim)
# print "fhpsi ",fhpsi.norm(),coeffs[1]
    Momentum.append(psi0.dot(psim))
# |psi_2>=2H|psi_1> - |psi_0>       
# |psi_m+1=2H|psi_m> - |psi_m-1>    
# or with other variables
# psimp1 = 2Hpsim - psi_mm1    

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=fhchebdeg).start()
    for nn in range(2,fhchebdeg):
        pbar.update(int(nn))
        psimm1.scale(-1.0)     
        Mat.mult(psim,psimp1)  
        psimp1.aypx(2.0*im,psimm1)
        fhpsi.axpy(1.0*coeffs[nn],psimp1)    #;print "fhpsi ",fhpsi.norm(),coeffs[1]
        
        psimm1=psim.copy()
        psim=psimp1.copy()
        Momentum.append(psi0.dot(psim))
 #print "fhpsi vege "
#print Momentum[-1]
    pbar.finish()
    return fhpsi,Momentum

def Chebyshev_rec(Mat,psi0,chebdeg):    
    Momentum=[]
 
# |psi0>=I|psi>
    psim=psi0.duplicate()  
    psimp1=psi0.duplicate()
    fhpsi=psi0.duplicate()
    psimm1=psi0.copy()

# |psi1>=H|psi>     
    Mat.mult(psi0,psim)
# print "psim ",psim.norm()

    #fhpsi.axpy(0.5,psimm1)
# print "fhpsi ",fhpsi.norm()
    Momentum.append(psi0.dot(psimm1))
 
    fhpsi.axpy(1.0,psim)
# print "fhpsi ",fhpsi.norm(),coeffs[1]
    Momentum.append(psi0.dot(psim))
# |psi_2>=2H|psi_1> - |psi_0>       
# |psi_m+1=2H|psi_m> - |psi_m-1>    
# or with other variables
# psimp1 = 2Hpsim - psi_mm1    

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=chebdeg).start()
    for nn in range(2,chebdeg):
        pbar.update(int(nn))
        psimm1.scale(-1.0)     
        Mat.mult(psim,psimp1)  
        psimp1.aypx(2.0,psimm1)
        fhpsi.axpy(1.0,psimp1)    #;print "fhpsi ",fhpsi.norm(),coeffs[1]
        
        psimm1=psim.copy()
        psim=psimp1.copy()
        Momentum.append(psi0.dot(psim))
        
 #print "fhpsi vege "
#print Momentum[-1]
    pbar.finish()
    return psim,Momentum

def Chebyshev_rec_arr(Mat,psi0,chebdeg):    
    Momentum=[]
    psi=[]
# |psi0>=I|psi>
    psim=psi0.duplicate()  
    psimp1=psi0.duplicate()
    fhpsi=psi0.duplicate()
    psimm1=psi0.copy()

# |psi1>=H|psi>     
    Mat.mult(psi0,psim)
# print "psim ",psim.norm()

    fhpsi.axpy(0.5,psimm1)
# print "fhpsi ",fhpsi.norm()
    Momentum.append(psi0.dot(psimm1))
    psi.append(fhpsi)
    
    fhpsi.axpy(1.0,psim)
# print "fhpsi ",fhpsi.norm(),coeffs[1]
    Momentum.append(psi0.dot(psim))
    psi.append(fhpsi)
# |psi_2>=2H|psi_1> - |psi_0>       
# |psi_m+1=2H|psi_m> - |psi_m-1>    
# or with other variables
# psimp1 = 2Hpsim - psi_mm1    

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=chebdeg).start()
    for nn in range(2,chebdeg):
        pbar.update(int(nn))
        psimm1.scale(-1.0)     
        Mat.mult(psim,psimp1)  
        psimp1.aypx(2.0,psimm1)
        fhpsi.axpy(1.0,psimp1)    #;print "fhpsi ",fhpsi.norm(),coeffs[1]
        
        psimm1=psim.copy()
        psim=psimp1.copy()
        Momentum.append(psi0.dot(psim))
 #print "fhpsi vege "
#print Momentum[-1]
    pbar.finish()
    return fhpsi,Momentum

def Timeprop(Mat, psi0, chebdeg, t):
 
 psim=psi0.duplicate()
 psimp1=psi0.duplicate()
 psit=psi0.duplicate()  
 psit.zeroEntries()     
 psimm1=psi0.copy()        # print "01psimm1 ",psimm1.norm(),jn(0,t)
    
# |psi_0> = psi0
# |psi_1> = -iH|psi_0>
 Mat.mult(psi0,psim)  
 psim.scale(-1.0j)      # print "01psim ",psim.norm(),jn(1,t)
 
# |psi_2> = J(0,t)|psi_0> + 2*J(1,t)|psi_1>
# |psi_i+1> = J(i-1,t)|psi_i-1> + 2*J(i,t)|psi_i>
 psit.axpy(1.0*jn(0,t),psimm1)  # print "01psit ",psit.norm()
 psit.axpy(2.0*jn(1,t),psim)    # print "02psit ",psit.norm() 
 for n in range(2,chebdeg):
  Mat.mult(psim,psimp1)
  psimp1.aypx(-2.0j,psimm1)
  psit.axpy(2.0*jn(n,t),psimp1) #  print "1psit ",psit.norm(),t,jn(n,t)
  
  psimm1=psim.copy()
  psim=psimp1.copy()
 return psit



def  LoadPetscVec(fname):
    viewer = PETSc.Viewer().createBinary(fname, 'r')
    Vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    Vec = PETSc.Vec().load(viewer)
    viewer.destroy() 
    Vec.setFromOptions()
    return Vec, Vec.getSize()

def  LoadPetscMat(fname):
    viewer = PETSc.Viewer().createBinary(fname, 'r')
    Mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    Mat = PETSc.Mat().load(viewer); viewer.destroy()
    dim = Mat.getSize(); Mat.setFromOptions()
    return Mat, dim


def RandomVec(dim, rndseed):

    Stime = np.datetime64(datetime.datetime.now()).item().microsecond
    if rndseed==-1:
        rndseed=int(Ftime%4+Stime%3)
        
    psir = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    psir.setSizes(dim)
    psir.setFromOptions()
    rctx = PETSc.Random().create(comm=PETSc.COMM_WORLD);
    rctx.setFromOptions()
    rctx.setInterval([-1.0-1.0j,1.0+1.0j])
    rctx.setSeed(rndseed)
    psir.setRandom(rctx)
    psir.normalize()
    Ftime = np.datetime64(datetime.datetime.now()).item().microsecond
    
    return  psir, Ftime
    
   
def OC_timeprop(H, psi0, N, args):    
#EZ MUKODIK SZTEM
    #//---------------------------------------------------
    #// Importing the Hamiltonian, and the two operators
    #// used in the correlation function. 
    #// THESE OBJECTS NEED TO BE LOADED !!!!!!
    #//---------------------------------------------------
    H, dim=LoadPetscMat(args.ham)
    OP1, dim=LoadPetscMat(args.op1)
    OP2, dim=LoadPetscMat(args.op2)
    fhcoeffs, dimfhcoeffs=LoadPetscVec(args.fhcoeffs)
    
    OUT=open(args.out+"-"+str(N),'w')
    #//Begining of the Chebyshev recursion
    #//Doing the recursion
    psitmp = psi0.copy()
    OP2.mult(psi0, psitmp)
    psit1,ChebMomentum = Chebyshev_F_rec(H,psitmp,args.fhchebdeg,fhcoeffs)
    psit2,ChebMomentum = Chebyshev_F_rec(H,psi0,args.fhchebdeg,fhcoeffs)
    psit1.axpy(-1, psitmp)
    tcorr=psit2.dot(psit1)
    
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=int(args.time/args.dt)+2).start()
    nn=0
    #//Doing the recursion
    for time in np.arange(0,args.time,args.dt):
        OUT.write("%f %f %f\n"%(time,tcorr.real,tcorr.imag))
        pbar.update(int(nn))
        nn+=1
        psitpdt1=Timeprop(H,psit1,args.chebdeg,args.dt)	#; print "1 ",psi1tpdt.norm()
        psitpdt2=Timeprop(H,psit2,args.chebdeg,args.dt)	#; print "1 ",psi1tpdt.norm()
        OP2.mult(psitpdt1, psitmp)
        tcorr=psitpdt2.dot(psitmp)	#; print "3 ",psitmp.norm(),tcorr
        psit1=psitpdt1.copy()
        psit2=psitpdt2.copy()
    
    OUT.close() 
    #TEMP.close() 
    pbar.finish()
    print "Chebyshev expansion done"
    
    psi0.destroy();       psitmp.destroy();   
    psit1.destroy();       psit2.destroy();   
    OP1.destroy();    OP2.destroy()
    

def Dos_timeprop(H, psi0, N, args):    
#EZ MUKODIK SZTEM
    #//---------------------------------------------------
    #// Importing the Hamiltonian, and the two operators
    #// used in the correlation function. 
    #// THESE OBJECTS NEED TO BE LOADED !!!!!!
    #//---------------------------------------------------
    fhcoeffs, dimfhcoeffs=LoadPetscVec(args.fhcoeffs)
    
    OUT=open(args.out+"-dos-"+str(N),'w')
    #//Begining of the Chebyshev recursion
    #//Doing the recursion
    psit1,ChebMomentum = Chebyshev_F_rec(H,psi0,args.fhchebdeg,fhcoeffs)
    tcorr=[]; 
    tcorr.append(psit1.dot(psi0)*Jackson(0, args.chebdeg+1))
    
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=int(args.time/args.dt)+1).start()
    nn=0
    print "Time propagation"
    #//Doing the recursion
    for time in np.arange(0, args.time,args.dt):
        pbar.update(int(nn))
        nn+=1
        OUT.write("%f %f %f\n"%(time,tcorr[-1].real,tcorr[-1].imag))
        psitpdt1=Timeprop(H,psit1,args.chebdeg,args.dt)	#; print "1 ",psi1tpdt.norm()
        tcorr.append(psitpdt1.dot(psi0)*Jackson(nn, args.chebdeg+1))	#; print "3 ",psitmp.norm(),tcorr
        psit1=psitpdt1.copy()

    OUT.close() 
    pbar.finish()
    print "Chebyshev expansion done"
    
    psi0.destroy();   
    psitpdt1.destroy(); psit1.destroy()
    
def QE_timeprop(H, psi0, N, args):    
    #//---------------------------------------------------
    #// Importing the Hamiltonian, and the two operators
    #// used in the correlation function. 
    #// THESE OBJECTS NEED TO BE LOADED !!!!!!
    #//---------------------------------------------------
    fhcoeffs, dimfhcoeffs=LoadPetscVec(args.fhcoeffs)
    
    OUT=open(args.out+"-dos-"+str(N),'w')
    #//Begining of the Chebyshev recursion
    #//Doing the recursion
    psit1,ChebMomentum = Chebyshev_F_rec(H,psi0,args.fhchebdeg,fhcoeffs)
    #psit1 = psi0.copy()
    tcorr=[]; 
    tcorr.append(psit1.dot(psi0))#*Jackson(0, args.chebdeg+1))
    
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=int(args.time/args.dt)+1).start()
    nn=0
    print "Time propagation"
    #//Doing the recursion
    for time in np.arange(0, args.time,args.dt):
        pbar.update(int(nn))
        nn+=1
        psitpdt1=Timeprop(H,psit1,args.chebdeg,args.dt)	#; print "1 ",psi1tpdt.norm()
        psit1=psitpdt1.copy()

	if nn%10==0:
		psitpdt1.scale(np.exp(1.0j*args.eps))
		psi=psitpdt1.array
		Han=np.hanning(len(psi))
		psi=psi*Han
		PPE=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi)))
		QE=open("hf-%d"%(nn),'w')
		sdim = 400
		for u in range(len(psi)):
			QE.write("%d %d %f %f\n"%(int(u/sdim),u%sdim,psi[u].real,psi[u].imag))
			if u%sdim==0:
			 QE.write("\n")
		QE.close()
		

    OUT.close() 
    pbar.finish()
    print "Chebyshev expansion done"
    
    psi0.destroy();   
    psitpdt1.destroy(); psit1.destroy()
    
    
def DC_timeprop(H, psi0, N, args):    
    #//---------------------------------------------------
    #// Importing the Hamiltonian, and the two operators
    #// used in the correlation function. 
    #// THESE OBJECTS NEED TO BE LOADED !!!!!!
    #//---------------------------------------------------
#    H, dim=LoadPetscMat(args.ham)
    OP1, dim=LoadPetscMat(args.op1)
    OP2, dim=LoadPetscMat(args.op2)
    fhcoeffs, dimfhcoeffs=LoadPetscVec(args.fhcoeffs)
    
    OUT=open(args.out+"-"+str(N),'w')
    #//Begining of the Chebyshev recursion
    #//Doing the recursion
    PSentmp = psi0.copy()
    PSentmp2 = psi0.copy()
    PSe = psi0.copy()
    psit = psi0.copy()

    fft_psi0=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi0)))/args.dt/np.pi;
    PSe.setValues( range(dim[0]), values=fft_psi0)
    PSen=PSe/(psi0.dot(PSe))
    OP2.mult(PSen, PSentmp)
    OP1.mult(psi0, psit)
    #tcorr=psit.dot(PSentmp)
    H.mult(PSen, PSentmp)
    tcorr=PSen.dot(PSentmp)
    
    #//Doing the recursion
    for time in np.arange(0,args.time,args.dt):
        OUT.write("%f %f %f\n"%(time,tcorr.real,tcorr.imag))

        psitpdt=Timeprop(H,psit,args.chebdeg,args.dt)	#; print "1 ",psi1tpdt.norm()
        fft_psitpdt=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psitpdt)))/args.dt/np.pi;
        PSe.setValues( range(dim[0]), values=fft_psitpdt)
    
        PSen=PSe/(psitpdt.dot(PSe))
        OP2.mult(PSen, PSentmp)
#        tcorr=psitpdt2.dot(psitmp)	#; print "3 ",psitmp.norm(),tcorr
        H.mult(PSen, PSentmp2)
        tcorr=PSen.dot(PSentmp)
        psit=psitpdt.copy()
    
    OUT.close() 


def const_polynom(E, chebdeg):
    coeffs=[]   
    for n in range(chebdeg):
        coeffs.append(np.cos(n*np.arccos(E)))
        
    return coeffs

def Ne(e):
    return 2./(np.pi*np.sqrt(1-e*e))
#    return 1

def ChebVec(E,cE,pE):
    nE=2*E*cE-pE
    return cE,nE
    
def Jackson(n, N):
    return ( (N-n)*np.cos(np.pi*n/N)+np.sin(np.pi*n/N)/np.tan(np.pi/N)) / N
    
def Mayoudos(H, dim, psi0, N, args):    
#EZ MUKODIK SZTEM
    rndseed=args.rndseed
#    print dim
    OUT=open(args.out+"-dosm",'w')
    ChebMomentum=[]
    for n in range(N):
        OUTK=open(args.out+"-kernel-"+str(n),'w')
        rndseed+=np.datetime64(datetime.datetime.now()).item().microsecond
        print rndseed
        psi0 = RandomVec(dim,rndseed)
        rndseed+=np.datetime64(datetime.datetime.now()).item().microsecond
        psir2 = RandomVec(dim, rndseed)
        psi0.axpy(1.0j,psir2)    
        psi0.normalize()  
        #psit, TempM = Chebyshev_rec(H,psi0,args.chebdeg)
        psit, TempM = Chebyshev_rec(H,psi0,args.chebdeg)
        ind=0
        for T in TempM:
            OUTK.write("%d %f %f\n"% (ind, T.real,T.imag))
            ind+=1
        OUTK.close()
        
        ChebMomentum.append(TempM)
        
    Ch=np.array(ChebMomentum)
    #Ch.reshape((N, args.chebdeg))
    Ch.reshape((N, len(Ch[0])))
    ChebMomentum=Ch.mean(axis=0)
    
    for E in np.arange(-0.999, 0.999, 0.01):
        cE=1
        nE=E
        Mom=[]
        MomJ=[]
        MomJ.append(Ne(E)*cE*ChebMomentum[0] * Jackson(0, args.chebdeg+1))
        MomJ.append(Ne(E)*nE*ChebMomentum[1] * Jackson(1, args.chebdeg+1))
        Mom.append(Ne(E)*cE*ChebMomentum[0] )
        Mom.append(Ne(E)*nE*ChebMomentum[1] )
        for i in range(1, args.chebdeg):
            cE, nE=ChebVec(E,nE,cE)
            if i==len(ChebMomentum):
                print "WARNING: i is greater than len(ChebMomentum)"
                break
            else:
                MomJ.append(Ne(E)*cE*ChebMomentum[i]*Jackson(i, args.chebdeg+1))
                Mom.append(Ne(E)*cE*ChebMomentum[i])
        
        MJ=np.sum(np.array(MomJ))
        M=np.sum(np.array(Mom))
#        OUT.write("%f %f %f\n"% (E, M.real,M.imag))
        OUT.write("%f %f %f\n"% (E, MJ.real,MJ.imag))
        
    OUT.close() 
  
def MayouQeig(H, dim, psi0, args):    

    psi0 = RandomVec(dim,args.rndseed+np.datetime64(datetime.datetime.now()).item().microsecond)
    psir2 = RandomVec(dim, args.rndseed+np.datetime64(datetime.datetime.now()).item().microsecond)
    psi0.axpy(1.0j,psir2)    
    psi0.normalize()  
    #psit, TempM = Chebyshev_rec(H,psi0,args.chebdeg)
    psit = psi0.copy()
    psitpdt=psi0.copy()
    
    E=0.1
    OUT=open(args.out+"-qeig-%3.2f"%(E),'w')
    cE=1
    nE=E
    psit.scale(Ne(E)*nE)
#    for i in range(args.chebdeg):
#    for i in range(10):
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=int(args.time/args.dt)+1).start()
    nn=0
    for time in np.arange(0,args.time,args.dt):
        psitpdt=Timeprop(H,psit,args.chebdeg,args.dt) 
        pbar.update(nn)
        nn+=1
        cE, nE=ChebVec(E,nE,cE)
        #J=Jackson(i, args.chebdeg+1)
        print nE
        J=1;
#        nE=1
        psit=psitpdt.copy()
        psit.scale(nE*J)
    
    psita=psit.array
    sdim=400
    for i in range(sdim):
        for j in range(sdim):
            A=psita[i*sdim+j]
            OUT.write("%d %d %f %f\n"% (i, j, A.real,A.imag))
        OUT.write(" \n")
        
        
    pbar.finish()
    OUT.close() 
    
def MayouDC(H, dim, psi0, args):    
    OP1, dim=LoadPetscMat(args.op1) 
    OP2, dim=LoadPetscMat(args.op2)
    rndseed=args.rndseed
    for N in range(args.nrss):
        psi0, rndseed=RandomVec(dim, rndseed)
        tmpMat1 = OP2.copy()
        tmpMat2 = OP2.copy()
        Momentum=[]
    
        O1psim1=psi0.duplicate()  
        O1psim2=psi0.duplicate()
        O2psim1=psi0.duplicate()  
        O2psim2=psi0.duplicate()
        O2psii_i=psi0.duplicate()
        psitmp=psi0.duplicate()
        O2psi0=psi0.duplicate()
        O1psi0=psi0.duplicate()
        
        OP1.mult(psi0, O1psi0)
    #    OP2.mult(psi0, O2psi0)
        
    #    O1psi0 = psi0.copy()
        ll=0
        tcorr=0.0
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=args.chebdeg).start()
        OUT=open(args.out+"-mdc-%d"%(N),'w')
        Mom=[]
        for nn in range(args.chebdeg):
            pbar.update(nn)
            if (nn == 0):		
                O1psi_i = O1psi0.copy()
                O1psim1 = O1psi_i.copy()
                
            elif (nn == 1):            
                H.mult(O1psi0,O1psi_i)
                O1psim2 = O1psim1.copy()
                O1psim1 = O1psi_i.copy()
            else:
                O1psim2.scale(-1.0)
                H.mult(O1psim1,O1psi_i)
                O1psi_i.aypx(2.0,O1psim2)
                O1psim2 = O1psim1.copy() 
                O1psim1 = O1psi_i.copy()
            
            OP2.mult(O1psi_i,psitmp)
    #        psitmp = O1psi_i.copy()
            O2psi0 = psi0.copy()
            O2psi_i = psi0.copy()
            for mm in range(args.chebdeg):
                if (mm == 0):
                    O2psi_i = O2psi0.copy()
                    O2psim1 = O2psi_i.copy()
                
                elif (mm == 1) :
                    H.mult(O2psi0,O2psi_i)
                    O2psim2 = O2psim1.copy()
                    O2psim1 = O2psi_i.copy()
            
                else:
                    O2psim2.scale(-1.0)
                    H.mult(O2psim1,O2psi_i)
                    O2psi_i.aypx(2.0,O2psim2)
                    O2psim2 = O2psim1.copy() 
                    O2psim1 = O2psi_i.copy()
            
                tcorr = psitmp.dot(O2psi_i)
                Mom.append(tcorr)
            
                OUT.write("%d %f %f\n"% (ll, tcorr.real,tcorr.imag)); ll+=1
        pbar.finish()
        OUT.close() 
        
        SF=open(args.out+"-sig-%d"%(N), 'w')


        for E in np.arange(-0.1, 0.1, 0.001):
            sig = 0.0
            ipE,  icE = 0, 1
            for i in range(args.chebdeg):
                if i==0:
                    ipE, icE=0, 1/2.
                if i==1:
                    ipE, icE=1, E
                
                for j in range(args.chebdeg):
                    if j==0:
                        jpE, jcE=0, 1/2.
                    if j==1:
                        jpE, jcE=1, E
                    
                    sig += icE*jcE*Mom[i*args.chebdeg+j]*Jackson(i, args.chebdeg)*Jackson(j, args.chebdeg)
                    jpE, jcE = ChebVec(E, jcE, jpE)
                    
                ipE, icE = ChebVec(E, icE, ipE)
            sig = sig*Ne(E)
            SF.write("%f %f %f\n"%(E, sig.real, sig.imag))
        
        SF.close()
    
    
    

def print_mat(H):
    print "     ", 
    N, M=H.shape
    for m in range(M):
        print "%2d"%(m+2), "  ", #*(3-len(str(m))), 
        
    print ""
    for n in range(N):
        print "%2d "%(n+2), 
        for m in range(M):
            print "%5.2f"%(H[n, m]),  
            
        print ""
