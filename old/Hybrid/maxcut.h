#ifndef MAXCUT_H
#define MAXCUT_H

#include "problem.h"

class MaxCutProblem:public Problem{
    public:
        MaxCutProblem(char* data_file);
        ~MaxCutProblem();
//        double neg_grad_largest_ev(double* a,double eta, double* new_u);
//        void uAu(double* new_u,double* new_uAu);
//        double uCu(double* new_u);
        // void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme) override;
};

#endif
