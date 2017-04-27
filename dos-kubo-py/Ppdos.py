import sys
import numpy as np

fname=sys.argv[1]
N=int(sys.argv[2])

Result=[]
dt=0
Han=[]
for nn in range(0,N):
 try:
  if N==1:
   Fdat=open(fname,'r').readlines()
  else:
   Fdat=open(fname+str(nn),'r').readlines()
 except IOError:
  1==1
 else:
  Dat=[]
  for L in Fdat:
   Dat.append([float(v) for v in L.split()])
 

#  Dat.reverse()
  Dat=np.array(Dat)

  tmpDat=Dat
  tmpDat=np.array([[-D[0],D[1],D[2]] for D in Dat[len(Dat):0:-1]] )
  Res=np.concatenate((tmpDat,Dat))
  newDat=Res[:,1]+Res[:,2]*1.0j
  
  T=Dat[:,0]; 
  if len(Han)==0:
   Han=np.hanning(len(newDat))

  
   
  newDat=newDat*Han
 
  if dt==0:
    dt=np.abs(T[1]-T[2])
   
  nf=1.0
  #fft(newDat)/dt/pi;
  fft_d=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(newDat)))/nf*dt/np.pi;
  w=np.linspace(-nf*np.pi/dt,nf*np.pi/dt,len(fft_d))
  w=np.array(w.T)
  if Result==[]:
   Result=fft_d
  else:
   Result+=fft_d
 
 
Result=Result/N
for wl,R in zip(w,Result):
 print wl,R.real,R.imag

 
 
