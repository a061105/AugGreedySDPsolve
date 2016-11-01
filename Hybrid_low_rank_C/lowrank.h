#ifndef LOWRANK_H
#define LOWRANK_H

#include "problem.h"

class LowRankProblem:public Problem{
    public:
        LowRankProblem(char* data_file);
        ~LowRankProblem();
//        double neg_grad_largest_ev(double* a,double eta, double* new_u);
//        void uAu(double* new_u,double* new_uAu);
//        double uCu(double* new_u);
        // void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme) override;
};

#endif
