133 /*------------------------------------------------------------------------/
      134 typedef struct primme_params {
      135 
      
      
      136 /  The user must input at least the following two arguments /
      137 int n;
      138 void (*matrixMatvec)
      
      
      
      139 ( void *x, void *y, int *blockSize, struct primme_params *primme);
      140 
      141 /  Preconditioner applied on block of vectors (if available) /
      142 void (*applyPreconditioner)
      143 ( void *x, void *y, int *blockSize, struct primme_params *primme);
      144 
      145 /  Matrix times a multivector for mass matrix B for generalized Ax = xBl /
      146 void (*massMatrixMatvec)
      147 ( void *x, void *y, int *blockSize, struct primme_params *primme);
      148 
      149 /  input for the following is only required for parallel programs /
      150 int numProcs;
      151 int procID;
      152 int nLocal;
      153 void *commInfo;
      154 void (*globalSumDouble)
      155 (void *sendBuf, void *recvBuf, int *count, struct primme_params *primme );
      156 
      157 /*Though primme_initialize will assign defaults, most users will set these */
158 int numEvals;  
159 primme_target target; 
160 int numTargetShifts; /  For targeting interior epairs,  /
    161 double *targetShifts; /  at least one shift must also be set /
    162 
    163 / the following will be given default values depending on the method */
    164 int dynamicMethodSwitch;
    165 int locking;
    166 int initSize;
    167 int numOrthoConst;
    168 int maxBasisSize;
    169 int minRestartSize;
    170 int maxBlockSize;
    171 int maxMatvecs;
    172 int maxOuterIterations;
    173 int intWorkSize;
    174 long int realWorkSize;
    175 int iseed[4];
    176 int *intWork;
    177 void *realWork;
    178 double aNorm;
    179 double eps;
    180 
    181 int printLevel;
    182 FILE *outputFile;
    183  
    184 void *matrix;
    185 void *preconditioner;
    186 double *ShiftsForPreconditioner;
    187 
    188 struct restarting_params restartingParams;
    189 struct correction_params correctionParams;
    190 struct primme_stats stats;
    191 struct stackTraceNode *stackTrace;
    192  
    193 } primme_params;
