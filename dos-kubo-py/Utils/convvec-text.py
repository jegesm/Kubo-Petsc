import sys
import numpy as np

fncoo = sys.argv[1]
Fcoo = open(fncoo,'r').readlines()[2:]

newtag = sys.argv[2]
fn = sys.argv[3]

ujfn=newtag+"_"+fn
ujF=open(ujfn,'w')
ujF.write("X Y Z V\n")  
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


fd = open(fn,'r').readlines()[2:]
for i in range(len(coords)):
    C=coords[i]
    L=fd[i].split()
    if len(L)>1:
        R,I = float(L[0]),float(L[2])
    else:
        R,I=0,0
    ujF.write("%f %f %f %f\n"%(C[0],C[1],C[2],R*R+I*I))
 #ujF.write("\n")

ujF.close() 
