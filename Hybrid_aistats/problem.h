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
    SparseMat C; // n-by-n matrix C
    vector<SparseMat2*> A; // m-by-n-by-n constraint matrices
    double* b;

    // not necessary, used in optimization.
//    double* prob_a;
//    double prob_eta;
    // eigenpairs of negative gradient. 
    // grad = C + eta*A^*(a)
    // new_u is the eigenvector, and eigenvalue is returned. 
    void neg_grad_largest_ev(double* a,double eta, double epsilon,int new_k,double* new_us, double* new_eigenvalues,int is_largest_eig);
    // new_uAu = A(new_u new_u^T)
    void uAu(double* new_u,double* new_uAu);
    // return <C,new_u new_U^T>
    double uCu(double* new_u);
    //virtual void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme) = 0;
    void non_convex_solver(vector<double*>& V, int K, double eta, double* y);
};

void set_prob(Problem *prob); 
void allocate_prob_a(int m); 

#endif
