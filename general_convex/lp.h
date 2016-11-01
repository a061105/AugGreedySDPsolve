#ifndef LP_H
#define LP_H

#include "problem.h"

void set_prob(Problem *prob); 

class LPProblem:public Problem{
    public:
        LPProblem(char* data_file);
        ~LPProblem();
        double neg_grad_largest_ev(double* a,double eta, double* new_u);
        void uAu(double* new_u,double* new_uAu);
        double uCu(double* new_u);
        // void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme) override;
};

#endif
