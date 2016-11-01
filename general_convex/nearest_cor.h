#ifndef NCM_H
#define NCM_H

#include "problem.h"

class NCMProblem:public Problem{
    public:
        NCMProblem(char* data_file);
	~NCMProblem();
	// the product between the gradient of lagrangian and a given vector.
	void gradientVecProd(double* xv,//input vector
			double* yv); //output vector
	// calculate the function value (returned) and the gradient of v
	double gradientV(const double* v, double *g);
	void solveSubProblemDiag(vector<double>& theta,int inner_iter_max);// diagonal terms, input for intial and output
	void update_y(); 
	void update_res_a(); 
	void infeasibilities(double& pinf, double& dinf, double& obj, double& dobj);
	double primal_obj();
	double primal_inf();
};

#endif
