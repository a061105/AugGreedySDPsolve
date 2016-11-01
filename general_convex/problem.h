#ifndef PROBLEM
#define PROBLEM

#include<vector>
#include "util.h"
extern "C" {
/* primme.h header file is required to run primme */
#include "primme.h"
}
using namespace std;

const int MAX_LINE = 10000000; //max char in a line in a file
const int FNAME_LENGTH = 256;//max length of a file name

typedef vector<pair<int,double> > SparseVec;
typedef map<int,double> SparseVec2;
typedef vector<SparseVec*> SparseMat;
typedef map<int,SparseVec2> SparseMat2;

/** This is a interface for problem aimed to utilize
 *  our optimization package. 
 */

class Problem{
	public:
		int n; // dimensionality
		int m; // number of constraint
		
		// Variables
		double* y; // dual variable m-by-1
		int rank; // rank of primal variable
		vector<double*> V; // n*rank-by-1

		double* res_a; // intermediate variable, res_a = A(X) - b+y/eta

		//used in optimization.
		double eta; // lagrangian constant/stepsize
		
		// the product between the gradient of lagrangian and a given vector.
		virtual void gradientVecProd(double* xv,//input vector
				double* yv){}; //output vector
		// calculate the function value (returned) and the gradient of v
		virtual double gradientV(const double* v, double *g){};
		virtual void solveSubProblemDiag(vector<double>& theta,int inner_iter_max){};// diagonal terms, input for intial and output
		virtual void update_y(){}; 
		virtual void update_res_a(){}; 
		virtual void infeasibilities(double& pinf, double& dinf, double& obj, double& dobj){};
//	private:
		virtual double primal_obj(){}; 
		virtual double primal_inf(){};
		// Problem
		SparseMat C; // n-by-n matrix C, for linear objective, <C,X>; for quadratic,\|X-C\|_F
		vector<SparseMat2*> A; // m-by-n-by-n constraint matrices
		double* b; // m-by-1 constraints
};

void set_prob(Problem *prob); 

#endif
