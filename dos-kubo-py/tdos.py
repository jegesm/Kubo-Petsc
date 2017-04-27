import sys
import numpy as np
fname=sys.argv[1]
Dat=open(fname,'r').readlines()

V=[]
for D in Dat:
 L=D.split()
 V.append([float(L[1]),float(L[2])])

N=len(V)

Xk=[np.cos(np.pi*(k+0.5)/N) for k in range(N)]

 
nef=open("proba",'w')

for i in range(N):
 T=0

 for j in range(N):
  T=T+(V[j][0]+1.0j*V[j][1])*np.cos(j*np.arccos(Xk[i]))
 RR=1/(np.pi*np.sqrt(1-Xk[i]*Xk[i]))*2*T
 nef.write("%d %f %f %f\n"%(i,Xk[i],np.real(RR),np.imag(RR)))
 

nef.close()
