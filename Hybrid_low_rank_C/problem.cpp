#include "problem.h"
#include <lbfgs.h>
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>
#include <random>
using namespace std;

double* prob_a;
double prob_eta;

static Problem *_prob;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);

void set_prob(Problem *prob) {_prob=prob;}
void allocate_prob_a(int m) {prob_a = new double[m];}

void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme){
    double* xv = (double*)x;        
    double* yv = (double*)y;
    for (int i=0;i<_prob->n;i++){
        yv[i] = 0.0;
    }
    for (int k=0;k<_prob->m;k++){
        SparseMat2* Ak = _prob->A[k];
        for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
            double tmp = 0.0;
            
            for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
                tmp += vit->second * xv[vit->first];
            }
            yv[it->first] += prob_a[k]*tmp;
        }
    }
    for (int i=0;i<_prob->n;i++){
	    yv[i] = -prob_eta * yv[i];
    } 
    double* tmpr = new double[_prob->VC.size()];
    for (int j=0;j<_prob->VC.size();j++){
	    double tmp = 0.0;
	    for (int ii=0;ii<_prob->n;ii++){
		    tmp += _prob->VC[j][ii]*xv[ii];
	    }
	    tmpr[j] = tmp;
    }
    for (int ii=0;ii<_prob->n;ii++){
	    double tmp=0;
	    for (int j=0;j<_prob->VC.size();j++){
		    tmp += tmpr[j]*_prob->VC[j][ii];
	    }
	    yv[ii] = -tmp;
    }


}


void Problem::neg_grad_largest_ev(double* a,double eta, double epsilon,int new_k,double* new_us, double* new_eigenvalues,int is_largest_eig){
    
    //double start = omp_get_wtime();
    for (int i=0;i<m;i++)
        prob_a[i] = a[i];
    prob_eta = eta;
    double *evals, *evecs, *rnorms;


    /* ----------------------------- */
    /* Initialize defaults in primme */
    /* ----------------------------- */
    primme_params primme;
    primme_preset_method method;
    method = DEFAULT_MIN_TIME;
    primme_initialize(&primme);

    /* ---------------------------------- */
    /* provide at least following inputs  */
    /* ---------------------------------- */
    primme.n = n;
    primme.eps = epsilon;
    primme.numEvals = new_k;
    primme.maxBasisSize = 35;
    primme.printLevel = 1;
    primme.matrixMatvec = gradVecProd;
    if (is_largest_eig==1){
        primme.target = primme_largest;
    }
    else if (is_largest_eig==0){
        primme.target = primme_smallest;
    }
    else {
        cerr<<"please specify 1 or 0 for is_largest_eig. 1: largest eigenvalue, 0: smallest eigenvalue"<<endl;
    }
    primme_set_method(method, &primme);
    primme.locking = 1;
    primme.initSize = 1;// primme.numEvals;
    /* Allocate space for converged Ritz values and residual norms */
    evals = (double *)primme_calloc(primme.numEvals, sizeof(double), "evals");
    evecs = (double *)primme_calloc(
            primme.n*primme.numEvals,sizeof(double), "evecs");
    rnorms = (double *)primme_calloc(primme.numEvals, sizeof(double), "rnorms");
    
    for(int i=0;i<primme.n * primme.numEvals;i++)
        evecs[i] = distribution(generator);
    /* ------------- */
    /*  Call primme  */
    /* ------------- */
    //double start2 = omp_get_wtime();
//    primme_display_params(primme);
    dprimme(evals, evecs, rnorms, &primme);

    //cerr << "eig solve time=" << omp_get_wtime()-start2 << endl;
    primme_Free(&primme);
    for (int i=0;i < primme.n * primme.numEvals;i++)
        new_us[i] = evecs[i];
    for (int i=0;i<primme.numEvals;i++){
        new_eigenvalues[i] = evals[i];
    }

    //cerr << "whole time=" << omp_get_wtime()-start << endl;
}
static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *v,
    lbfgsfloatval_t *g,
    const int N,
    const lbfgsfloatval_t step
    )
{
    double* y = *(static_cast<double**>(instance));
    lbfgsfloatval_t fx = 0.0;

    int K = N/_prob->n;
    
    for (int ii=0;ii<N;ii++)
	    g[ii] = 0.0;
	
    double* aa = new double[_prob->m];

    int m = _prob->m;
    // form a=A(UU')-b+y/eta
    for (int k=0;k<m;k++){
	    SparseMat2* Ak = _prob->A[k];
	    aa[k]=0;
	    for (int i=0;i<K;i++){
		    int shift_index =  _prob->n*i;
		    for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
			    double tmp = 0.0;
			    for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
				    tmp += vit->second * v[shift_index+vit->first];
			    }
			    aa[k] += tmp * v[shift_index+it->first];
		    }
	    }
	    aa[k] = aa[k] - _prob->b[k] + y[k]/prob_eta;
	    fx += aa[k]*aa[k];
    }
    fx *= (prob_eta/2);
    for (int k=0;k<m;k++){
    	    SparseMat2* Ak = _prob->A[k];
	    for (int i=0;i<K;i++){
		    int shift_index =  _prob->n*i;
		    double* gi = g + _prob->n*i;
		    for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
			    double tmp = 0.0;
			    for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
				    tmp += vit->second * v[shift_index+vit->first];
			    }
			    gi[it->first] += 2*prob_eta*tmp*aa[k];
		    }
	    }
    }
    // C part
    double* tmpr = new double[_prob->VC.size()];
    for (int i=0;i<K;i++){
	    int shift_index =  _prob->n*i;
	    double* gi = g + _prob->n*i;
	    
	    for (int j=0;j<_prob->VC.size();j++){
		    double tmp = 0.0;
		    for (int ii=0;ii<_prob->n;ii++){
		    	tmp += _prob->VC[j][ii]*v[shift_index+ii];
		    }
		    tmpr[j] = tmp;
	    }
	    for (int ii=0;ii<_prob->n;ii++){
		    double tmp=0;
		    for (int j=0;j<_prob->VC.size();j++){
		    	tmp += tmpr[j]*_prob->VC[j][ii];
		    }
		    gi[ii] = 2*tmp;
		    fx += tmp*v[shift_index+ii];
	    }
    }
    delete[] tmpr;
    delete[] aa;
    return fx;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    //printf("Iteration %d,  fx = %f, gnorm = %f, step = %f\n", k, fx, gnorm, step);
    return 0;
}

void Problem::non_convex_solver(vector<double*>& V, int K, double eta, double* y){
    int N=n*K;
    prob_eta = eta;
    int ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N);
    /* Initialize the variables. */
    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return;
    }

    for (int i=0;i<K;i++){
	    for (int j=0;j<n;j++){
		    int ii = i*n+j;
		    x[ii] = V[i][j];
	    }
    }
    lbfgs_parameter_t param;
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.max_iterations = 100;
    param.m = 10;
    // param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
    param.epsilon = 1e-3;//LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
    //param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */
    ret = lbfgs(N, x, &fx, evaluate, progress, &y, &param);
 
    for (int i=0;i<K;i++){
	    for (int j=0;j<n;j++){
		    int ii = i*n+j;
		    V[i][j] = x[ii];
	    }
    }
    /* Report the result. */
    //printf("L-BFGS optimization terminated with status code = %d\n", ret);
    //exit(0); 
    lbfgs_free(x);	
}


void Problem::uAu(double* new_u,double* new_uAu){
    for (int k=0;k<m;k++){
        double uAuk = 0.0;
        SparseMat2* Ak = A[k];
        for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
            double tmp = 0.0;
            for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
                tmp += vit->second * new_u[vit->first];
            }
            uAuk += tmp * new_u[it->first];
        }
        new_uAu[k] = uAuk;
    }   
}

double Problem::uCu(double* new_u){
    double uCuv = 0.0;
    for (int i=0;i<VC.size();i++){
        double tmp = 0.0;
	for (int j=0;j<n;j++){
		tmp += new_u[j]*VC[i][j];
	}
	uCuv += tmp*tmp;
    }
    return uCuv;
}

