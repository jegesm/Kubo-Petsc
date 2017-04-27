import sys
import numpy as np

#COORDS
fncoo = sys.argv[1]
Fcoo = open(fncoo,'r').readlines()[2:]
dim=len(Fcoo)
coords=np.zeros((dim, 3))
for LL in Fcoo:
    LL = LL.replace("(","")
    LL = LL.replace(")","")
    LL = LL.replace(",","")
    LL = LL.replace(":","")
    L=LL.split()
    row=int(L[1])
    coords[row]=[float(L[3]), float(L[5]), float(L[7])]

fham = sys.argv[2]
Fd = open(fham,'r').readlines()[2:]
ham=np.zeros((dim, dim))
for LL in Fd:
    LL = LL.replace("(","")
    LL = LL.replace(")","")
    LL = LL.replace(",","")
    LL = LL.replace(":","")
    L=LL.split()
    row=int(L[1])
    for i in range(len(L[2:])/2):
        col=int(L[2+i*2])
        V=L[3+i*2]
        ham[row, col]=V
        


ujfn="mat_"+fham
ujF=open(ujfn,'w')
ujF.write("# X Y V\n")  
nonz=np.nonzero(ham)
nonzz=zip(nonz[0], nonz[1])
for i, j in nonzz:
    ujF.write("%d %d %f\n"%(i, j, ham[i, j]))
ujF.close() 

ujfn="cmat_"+fham
ujF=open(ujfn,'w')
ujF.write("# X Y Z V\n")  
nonz=np.nonzero(ham)
nonzz=zip(nonz[0], nonz[1])
for i, j in nonzz:
    ujF.write("%f %f %f %f\n"%(coords[i][0], coords[i][1], coords[i][2], ham[i, j]))
ujF.close() 
