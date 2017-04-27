include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules


test: test.o fhpsi.o
	-${CLINKER} -o test test.o  fhpsi.o ${PETSC_MAT_LIB}
	${RM} test.o fhpsi.o

AC_KUBO: AC_KUBO.o timeprop_MK2.o fhpsi.o chkopts
	-${CLINKER} -o AC_KUBO AC_KUBO.o fhpsi.o timeprop_MK2.o  ${PETSC_MAT_LIB}
	${RM} AC_KUBO.o fhpsi.o timeprop_MK2.o 


include ${PETSC_DIR}/conf/test
