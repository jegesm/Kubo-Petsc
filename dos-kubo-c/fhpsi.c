//Chebyshev expansion of an arbitrary function of H
//Chebyshev coeffitients are stored in the vector coeff
//|fhpsi>=f(H)|psi0>
//chedeg controls the order of the expansion
#include <petscksp.h>
PetscErrorCode fhpsi(Mat *H,Vec psi0,Vec fhpsi,PetscInt chebdeg,Vec coeff)
{

Vec psim1,psim2,psii; 
PetscErrorCode ierr;
PetscInt i,ix[chebdeg];
PetscScalar   one=1.0;
PetscScalar   ck[chebdeg];




for (i=0; i<chebdeg; i++) {
ix[i]=i;
}

ierr=VecGetValues(coeff,chebdeg,ix,ck);CHKERRQ(ierr); 


VecZeroEntries(fhpsi); 
//Initialize workspace vectors
ierr = VecDuplicate(psi0,&psim1);CHKERRQ(ierr); 
ierr = VecDuplicate(psi0,&psim2);CHKERRQ(ierr);
ierr = VecDuplicate(psi0,&psii);CHKERRQ(ierr);
PetscPrintf(PETSC_COMM_WORLD,"#Chebyshev expansion start. \n");  

//Initialize Chebysev recursion
 VecCopy(psi0,psim2);                            //psi_m2=psi0
 MatMult(*H,psi0,psim1);                         //psi_m1=H*psi0
 VecAXPY(fhpsi,one/2.0*ck[0],psim2);              //fhpsi=ck(0)/2*psi_m2+ck(1)*psi_m1;
 VecAXPY(fhpsi,ck[1],psim1);                      
 //Doing the recursion

//PetscPrintf(PETSC_COMM_WORLD,"Iteration %d %e\r",0,ck[0]);  
//PetscPrintf(PETSC_COMM_WORLD,"Iteration %d %e\r",1,ck[1]);  

 for (i=2; i<chebdeg; i++) {
 VecScale(psim2,-one);
 MatMult(*H,psim1,psii); VecAYPX(psii,2.0*one,psim2);//psi_i=2.0*H*psi_m1-psi_m2;
 VecAXPY(fhpsi,ck[i],psii);                          //fhpsi=fhpsi+ck(i)*psi_i;     

 VecCopy(psim1,psim2); 
 VecCopy(psii,psim1);

//PetscPrintf(PETSC_COMM_WORLD,"Iteration %d %e\r",i,ck[i]);  

 }
PetscPrintf(PETSC_COMM_WORLD,"#Chebyshev expansion done.\n");
//Destroy workspace
ierr = VecDestroy(&psim1);CHKERRQ(ierr);
ierr = VecDestroy(&psim2);CHKERRQ(ierr);
ierr = VecDestroy(&psii);CHKERRQ(ierr);
return(0);  
}


