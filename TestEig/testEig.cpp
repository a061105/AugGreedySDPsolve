#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <math.h>

extern "C"
{
/* primme.h header file is required to run primme */
#include "primme.h"

/* wtime.h header file is included so primme's timimg functions can be used */
#include "wtime.h"
}
// multiply to an Identity Matrix
void matVecProd(void* x, void* y, int* blockSize, primme_params* primme){
	
	double* xv = (double*)x;
	double* yv = (double*)y;
	int n = primme->n;
	for(int i=0;i<n;i++){
		yv[i] = i*xv[i];
	}
}

int main(){
	
   int ret;
   /* Timing vars */
   double ut1,ut2,st1,st2,wt1,wt2;
	
   double *evals, *evecs, *rnorms;
   
   /* ----------------------------- */
   /* Initialize defaults in primme */
   /* ----------------------------- */
   primme_params primme;
   primme_preset_method method;
   method = DYNAMIC;
   primme_initialize(&primme);

   /* ---------------------------------- */
   /* provide at least following inputs  */
   /* ---------------------------------- */
   primme.n = 10;
   primme.eps = 1e-3;
   primme.numEvals = 5;
   primme.printLevel = 2;
   primme.matrixMatvec = matVecProd;
   primme_set_method(method, &primme);
   
   /* Allocate space for converged Ritz values and residual norms */
   evals = (double *)primme_calloc(primme.numEvals, sizeof(double), "evals");
   evecs = (double *)primme_calloc(
		primme.n*primme.numEvals,sizeof(double), "evecs");
   rnorms = (double *)primme_calloc(primme.numEvals, sizeof(double), "rnorms");

   /* ------------- */
   /*  Call primme  */
   /* ------------- */

   wt1 = primme_get_wtime(); 
   primme_get_time(&ut1,&st1);
   ret = dprimme(evals, evecs, rnorms, &primme);

   for (int t=0;t<primme.numEvals;t++){
       for (int j=0;j<primme.n;j++) 
           printf("%f ", evecs[t*primme.n+j]);
       printf("\n");
   }
   wt2 = primme_get_wtime();
   primme_get_time(&ut2,&st2);

   /* Reporting */
   printf("Wallclock Runtime   : %-f\n", wt2-wt1);
   printf("User Time           : %f seconds\n", ut2-ut1);
   printf("Syst Time           : %f seconds\n", st2-st1);

   primme_Free(&primme);

   return(ret);

}
