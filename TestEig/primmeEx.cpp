329 /****************************************************************************
330 * Applies the matrix vector multiplication on a block of vectors.
331 * Because a block function is not available, we call blockSize times
332 * the SPARSKIT function amux(). Note the (void ) parameters x, y that must 
333 * be cast as doubles for use in amux()
334 *
335 ******************************************************************************/
336 void MatrixMatvec(void *x, void *y, int *blockSize, primme_params *primme) {
337  
338 int i;
339 double *xvec, *yvec;
340 CSRMatrix *matrix;
341 //double tempo[70000], shift=-13.6;
342  
343 matrix = (CSRMatrix *)primme->matrix;
344 xvec = (double *)x;
345 yvec = (double *)y;
346 
347 for (i=0;i<*blockSize;i++) {
348 amux_(&primme->n, &xvec[primme->nLocal*i], &yvec[primme->nLocal*i], 
349 matrix->AElts, matrix->JA, matrix->IA);
350 
351 // Brute force implementing (A-sigme)^2
352 // yvec = ax tempo = aax
353 // y=(a-sI)(a-sI)x=(aa+ss-2sa)x = tempo + ss*x -2s(yvec). 
354 /
355 amux_(&primme->n, &yvec[primme->nLocal*i], tempo,
356 matrix->AElts, matrix->JA, matrix->IA);
357 Num_axpy_dprimme(primme->n, shift*shift, &xvec[primme->nLocal*i], 1, tempo, 1);
358 Num_axpy_dprimme(primme->n, -2*shift, &yvec[primme->nLocal*i], 1, tempo, 1);
359 Num_dcopy_primme(primme->n, tempo, 1, &yvec[primme->nLocal*i],1);
360 */
361 }
362 
363 }
