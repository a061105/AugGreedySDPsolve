#ifndef MAXCUT_H
#define MAXCUT_H

#include "problem.h"

class MaxCutProblem:public Problem{
    public:
        MaxCutProblem(char* data_file);
	~MaxCutProblem();
	// the product between the gradient of lagrangian and a given vector.
	void gradientVecProd(double* xv,//input vector
			double* yv); //output vector
	// calculate the function value (returned) and the gradient of v
	double gradientV(const double* v, double *g);
	void solveSubProblemDiag(double* theta);// diagonal terms, input for intial and output
	void update_y(); 
	void update_res_a(); 
	void infeasibilities(double& pinf, double& dinf, double& obj, double& dobj);
	double primal_obj();
	double primal_inf();
};

#endif
