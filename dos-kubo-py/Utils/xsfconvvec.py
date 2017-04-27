import sys

xd = int(sys.argv[1])
yd = int(sys.argv[2])
zd = int(sys.argv[3])
fn = sys.argv[4]
fd = open(fn,'r').readlines()[2:]

if len(fd)!=xd*yd*zd:
  print "HMM"

ujfn="p"+fn
ujF=open(ujfn,'w')
ujF.write("""
 CRYSTAL
 PRIMVEC
    %14.10f 0.000000000    0.000000000
    0.000000000    %14.10f    0.000000000
    0.000000000    0.000000000   %14.10f
 PRIMCOORD
           2           1
C         1.231139480    0.710798710    6.000679830
C         0.000000000    1.421597420    6.000679830
""" %(cell[0],cell[4],cell[9])

ujF.write("""BEGIN_BLOCK_DATAGRID_3D
3D_PWSCF
DATAGRID_3D_UNKNOWN
         100         100         100
  0.000000  0.000000  0.000000
  %14.10f  0.000000  0.000000
  0.00000  %14.10f   0.000000
  0.000000  0.000000 %14.10f

""" %(cell[0],cell[4],cell[9])


for x in range(xd):
 for y in range(yd):
  for z in range(zd):
   R,I = float(fd[x*zd*yd+y*zd+z].split()[0]),float(fd[x*zd*yd+y*zd+z].split()[2])
   ujF.write("%f\n"%(x,y,z,R*R+I*I))
 ujF.write("\n")
 
