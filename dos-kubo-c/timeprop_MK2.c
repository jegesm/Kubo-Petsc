//The time propagation function using Chebyshev expansion of the propagator
#include <petscksp.h>
PetscErrorCode timeprop_MK2(Mat *H,Vec psi0,Vec psit,PetscInt chebdeg,PetscReal t)
{

Vec psim1,psim2,psii; 
PetscErrorCode ierr;
PetscInt i;
PetscScalar   one=1.0;

VecZeroEntries(psit); 
//Initialize workspace vectors
ierr = VecDuplicate(psi0,&psim1);CHKERRQ(ierr); 
ierr = VecDuplicate(psi0,&psim2);CHKERRQ(ierr);
ierr = VecDuplicate(psi0,&psii);CHKERRQ(ierr);
//jn are Bessel functions of the first kind 
//they are the Chebyshev coefficients for exp(x)
//Initialize Chebysev recursion
 VecCopy(psi0,psim2);                            //psi_m2=psi0

 MatMult(*H,psi0,psim1);VecScale(psim1,-PETSC_i);//psi_m1=-1.0i*H*psi0
 VecAXPY(psit,one*jn(0,t),psim2);                //psit=jn(0,t)*psi_m2+2*jn(1,t)*psi_m1;
 VecAXPY(psit,2*one*jn(1,t),psim1);                      
 //Doing the recursion
 for (i=2; i<chebdeg+1; i++) {
 MatMult(*H,psim1,psii); VecAYPX(psii,-2.0*PETSC_i,psim2);//psi_i=-2.0i*H*psi_m1+psi_m2;
 VecAXPY(psit,2*one*jn(i,t),psii);              //psit=psit+2*jn(i,t)*psi_i;     
 VecCopy(psim1,psim2); 
 VecCopy(psii,psim1);
 }

//Destroy workspace
ierr = VecDestroy(&psim1);CHKERRQ(ierr);
ierr = VecDestroy(&psim2);CHKERRQ(ierr);
ierr = VecDestroy(&psii);CHKERRQ(ierr);
return(0);  
}


