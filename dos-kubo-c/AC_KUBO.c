static char help[] = "Simple time dependent Schrodinger equation based evaluation for correlation functions. \n\
-rndseed random seed for the random starting vector \n\
-dt defines the time step \n\
-T defines the total time of propagation \n\
-chebdeg defines degree of Chebyshev expansion in time evolution\n\
-outfname output file name for the realtime correlation function\n\
-hamfile name of file with the hamiltonian                                 (MUST SUPPLY)\n\
-op1 name of file with the first operator                                  (MUST SUPPLY)\n\
-op2 name of file with the second operator                                 (MUST SUPPLY)\n\
-fcoeff name of file with the Chebyshev coefficients of the Fermi function (MUST SUPPLY)\n\
-fhchebdeg defines degree of Chebyshev expansion in the evaluation of f(H) \n";

#include <petscksp.h>

extern PetscErrorCode timeprop_MK2(Mat *,Vec,Vec,PetscInt,PetscReal);
extern PetscErrorCode fhpsi(Mat *H,Vec psi0,Vec fhpsi,PetscInt chebdeg,Vec coeff);

int main(int argc,char **args)
{
//  PetscErrorCode ierr; 
//  PetscInt       i;
  PetscInitialize(&argc,&args,(char *)0,help);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This program requires complex numbers");
#endif

// Setting output file
PetscViewer    viewer;
PetscBool flg;
char      filename[PETSC_MAX_PATH_LEN];     
PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
PetscViewerSetType(viewer, PETSCVIEWERASCII);
PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
strcpy(filename, "dat");
PetscOptionsGetString(PETSC_NULL,"-outfname",filename,PETSC_MAX_PATH_LEN,&flg);
PetscViewerFileSetName(viewer, filename);

//---------------------------------------------------
// Importing the Hamiltonian, Fermi function Chebyshev coeffitients  
// and the two operators used in the correlation function 
// THESE OBJECTS NEED TO BE LOADED !!!!!!
//---------------------------------------------------

// Import coefficient vector for Chebyshev expansion of the Fermi function
Vec            ccffv;
PetscViewer           fcv;
PetscBool             fcflg;
char      fcfile[PETSC_MAX_PATH_LEN];
VecCreate(PETSC_COMM_WORLD,&ccffv);
PetscViewerCreate(PETSC_COMM_WORLD, &fcv);
PetscOptionsGetString(PETSC_NULL,"-fcoeff",fcfile,PETSC_MAX_PATH_LEN,&fcflg);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,fcfile,FILE_MODE_READ,&fcv);
VecLoad(ccffv,fcv);
PetscViewerDestroy(&fcv);
VecAssemblyBegin(ccffv);   VecAssemblyEnd(ccffv);

Vec            ccffv_SEQ;
VecScatter     ctx;
VecScatterCreateToAll(ccffv,&ctx,&ccffv_SEQ);
VecScatterBegin(ctx,ccffv,ccffv_SEQ,INSERT_VALUES,SCATTER_FORWARD);
VecScatterEnd(ctx,ccffv,ccffv_SEQ,INSERT_VALUES,SCATTER_FORWARD);
VecScatterDestroy(&ctx);

// Importing the Hamiltonian
Mat H;
PetscViewer           hamfv;
PetscBool             hflg;
char      hamfile[PETSC_MAX_PATH_LEN];
PetscViewerCreate(PETSC_COMM_WORLD, &hamfv);
PetscOptionsGetString(PETSC_NULL,"-hamfile",hamfile,PETSC_MAX_PATH_LEN,&hflg);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,hamfile,FILE_MODE_READ,&hamfv);
MatCreate(PETSC_COMM_WORLD,&H);
MatLoad(H,hamfv);
PetscViewerDestroy(&hamfv);
PetscInt       dim;
MatGetSize(H,0,&dim);
MatSetFromOptions(H);

// Importing OP1
Mat OP1;
PetscViewer           op1fv;
PetscBool             op1flg;
char      op1file[PETSC_MAX_PATH_LEN];
PetscViewerCreate(PETSC_COMM_WORLD, &op1fv);
PetscOptionsGetString(PETSC_NULL,"-op1",op1file,PETSC_MAX_PATH_LEN,&op1flg);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,op1file,FILE_MODE_READ,&op1fv);
MatCreate(PETSC_COMM_WORLD,&OP1);
MatLoad(OP1,op1fv);
PetscViewerDestroy(&op1fv);
MatSetFromOptions(OP1);

// Importing OP2
Mat OP2;
PetscViewer           op2fv;
PetscBool             op2flg;
char      op2file[PETSC_MAX_PATH_LEN];
PetscViewerCreate(PETSC_COMM_WORLD, &op2fv);
PetscOptionsGetString(PETSC_NULL,"-op2",op2file,PETSC_MAX_PATH_LEN,&op2flg);
PetscViewerBinaryOpen(PETSC_COMM_WORLD,op2file,FILE_MODE_READ,&op2fv);
MatCreate(PETSC_COMM_WORLD,&OP2);
MatLoad(OP2,op2fv);
PetscViewerDestroy(&op2fv);
MatSetFromOptions(OP2);
// H is the Hamiltonian OP1 and OP2 are the two operators
//---------------------------------------------------
//---------------------------------------------------
// Initiate random vector
//---------------------------------------------------
Vec psir,psii;
PetscRandom    rctx;
VecCreate(PETSC_COMM_WORLD,&psir);
VecSetSizes(psir,PETSC_DECIDE,dim);
VecSetFromOptions(psir);
VecDuplicate(psir,&psii);

PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
PetscRandomSetFromOptions(rctx);
PetscRandomSetInterval(rctx,-1.0,1.0);

PetscInt seed=0;PetscOptionsGetInt(PETSC_NULL,"-rndseed",&seed,PETSC_NULL);
PetscRandomSetSeed(rctx,seed); PetscRandomSeed(rctx);

VecSetRandom(psir,rctx);VecSetRandom(psii,rctx);
VecAXPY(psir,PETSC_i,psii);

PetscReal  norm;
VecNorm(psir, NORM_2,&norm);
VecNormalize(psir,&norm);

Vec psi0;
VecDuplicate(psir,&psi0); VecCopy(psir,psi0);
VecDestroy(&psir);     VecDestroy(&psii);
// Random vector initialized and is in psi0
//---------------------------------------------------
//---------------------------------------------------
//Initiating vectors used in the correlator by Chebyshev expansion
//---------------------------------------------------
PetscPrintf(PETSC_COMM_WORLD,"#Initialization of starting vectors\n");   
PetscInt fhchebdeg=220;PetscOptionsGetInt(PETSC_NULL,"-fhchebdeg",&fhchebdeg,PETSC_NULL);
// Calculate |psi2>=f(H)|psi0> by Chebyshev expansion
Vec psi2;VecDuplicate(psi0,&psi2);
fhpsi(&H,psi0,psi2,fhchebdeg,ccffv_SEQ);
// Calculate |psi1>=(1-f(H))OP2|psi0> by Chebyshev expansion
Vec psi1,psi1tmp;VecDuplicate(psi0,&psi1);VecDuplicate(psi0,&psi1tmp);
MatMult(OP2,psi0,psi1); //psi1=OP2*psi0
fhpsi(&H,psi1,psi1tmp,fhchebdeg,ccffv_SEQ);//psi1tmp=f(H)*psi1=f(H)*OP2*psi0

PetscScalar    one = 1.0;
VecAXPY(psi1,-one,psi1tmp);VecDestroy(&psi1tmp);
//psi1=psi1-psi1tmp=OP2*psi0-f(H)*OP2*psi0=(1-f(H))*OP2*psi0
PetscPrintf(PETSC_COMM_WORLD,"#Initialization of starting vectors DONE\n");   
//---------------------------------------------------
// Initializing dummy vectors for time propagation
//---------------------------------------------------
Vec psitmp;
Vec psi1t,psi1tpdt;
Vec psi2t,psi2tpdt;
VecDuplicate(psi0,&psitmp);
VecDuplicate(psi0,&psi1t);
VecDuplicate(psi0,&psi2t);
VecDuplicate(psi0,&psi1tpdt);
VecDuplicate(psi0,&psi2tpdt);

VecCopy(psi1,psi1t);VecCopy(psi2,psi2t);
VecDestroy(&psi1);     VecDestroy(&psi2);  
//---------------------------------------------------
// Propagating the initial states psi1t and psi2t
// and calculating the correlation function
//---------------------------------------------------
// Set propagation parameters
PetscReal time,dt=3,T=10;
PetscInt chebdeg=20;
// Getting options from command line 
PetscOptionsGetReal(PETSC_NULL,"-dt",&dt,PETSC_NULL);
PetscOptionsGetReal(PETSC_NULL,"-T",&T,PETSC_NULL);
PetscOptionsGetInt(PETSC_NULL,"-chebdeg",&chebdeg,PETSC_NULL);
PetscScalar   tcorr; //This variable contains the value of the correlator
// output at t=0
time=0.0;tcorr=1.0;
MatMult(OP1,psi1t,psitmp); VecDot(psi2t,psitmp,&tcorr);
PetscViewerASCIIPrintf(viewer, "%e %e %e \n",time,PetscRealPart(tcorr),PetscImaginaryPart(tcorr));
// propagation is done here
PetscPrintf(PETSC_COMM_WORLD,"#Time propagation starts\n");   
for (time=dt; time<T+dt; time=time+dt) {
// PetscPrintf(PETSC_COMM_WORLD,"Time  %e \r",time);   
 timeprop_MK2(&H,psi1t,psi1tpdt,chebdeg,dt);
 timeprop_MK2(&H,psi2t,psi2tpdt,chebdeg,dt);
 // Calculating the timecorrelation function
 MatMult(OP1,psi1tpdt,psitmp); VecDot(psi2tpdt,psitmp,&tcorr);
 PetscViewerASCIIPrintf(viewer, "%e %e %e \n",time,PetscRealPart(tcorr),PetscImaginaryPart(tcorr));
 // New is old
 VecCopy(psi1tpdt,psi1t);
 VecCopy(psi2tpdt,psi2t);
}
PetscPrintf(PETSC_COMM_WORLD,"#Time propagation done.               \n");   
//---------------------------------------------------
// Finalize
//---------------------------------------------------
VecDestroy(&ccffv);    VecDestroy(&psi0); VecDestroy(&ccffv_SEQ);
VecDestroy(&psi1tpdt); VecDestroy(&psi1t);  

VecDestroy(&psi2tpdt); VecDestroy(&psi2t);  
VecDestroy(&psitmp);

MatDestroy(&H);MatDestroy(&OP1);MatDestroy(&OP2);

PetscFinalize();
return 0;
}

