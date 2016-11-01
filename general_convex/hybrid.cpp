#include<math.h>
#include <lbfgs.h>
#include <random>
#include <time.h>
#include<vector>
#include<cstring>
#include<stdlib.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<deque>
#include <ctime>
#include <iomanip>  
#include <algorithm>
#include "problem.h"
#include "maxcut.h"
#include "nearest_cor.h"
#include "lp.h"

using namespace std;

class Param{
	public:
	Param(){
		problem_type = 1;
		inner_iter = 30;
		outer_iter = 100;
		yt_iter = 100;
		eta = 0.1;
	}
	int problem_type;
	int inner_iter;
	int outer_iter;
	int yt_iter;
	double eta;
};

Param param;
void exit_with_help(){

    cerr << "Usage: ./sdp_omp (options) [train_data] (model)" << endl;
    cerr << "options:" << endl;
    cerr << "-p problem_type: (default 1)" << endl;
    cerr << "	1 -- max cut" << endl;
    cerr << "	2 -- nearest correlation matrix" << endl;
    cerr << "	3 -- linear programming" << endl;
    cerr << "-t step size (default 1)" << endl;
    cerr << "-e eta (default 0.1)" << endl;
    cerr << "-i number of inner iterations (default 30)" << endl;
    cerr << "-o number of outer iterations (default 100)" << endl;
    cerr << "-y number of yt iterations (default 100)" << endl;
     
    //cerr << "-m max_iter: maximum_outer_iteration (default 100)" << endl;
    //cerr << "-e epsilon: stop criterion (default 1e-6)"<<endl;
    exit(0);
}

void parse_command_line(int argc, char** argv, char*& train_file, char*& model_file){

    int i;
    for(i=1;i<argc;i++){

        if( argv[i][0] != '-' )
            break;
        if( ++i >= argc )
            exit_with_help();

        switch(argv[i-1][1]){

            case 'p': param.problem_type = atoi(argv[i]);
                      break;
            case 'i': param.inner_iter = atoi(argv[i]);
                      break;
            case 'o': param.outer_iter = atoi(argv[i]);
                      break;
            case 'y': param.yt_iter = atoi(argv[i]);
                      break;
            case 'e': param.eta = atof(argv[i]);
                      break;
            default:
                      cerr << "unknown option: -" << argv[i-1][1] << endl;
        }
    }

    if(i>=argc)
        exit_with_help();

    train_file = argv[i];
    i++;

    if( i<argc )
        model_file = argv[i];
    else
        strcpy(model_file,"model");
}


static Problem *_prob;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);

void set_prob(Problem *prob) {_prob=prob;}

void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme){
    double* xv = (double*)x;        
    double* yv = (double*)y;
    _prob->gradientVecProd(xv,yv);
}


void neg_grad_largest_ev(double epsilon,int new_k,double* new_us, double* new_eigenvalues,int is_largest_eig){	
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
    primme.n = _prob->n;
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
//    primme_display_params(primme);
    dprimme(evals, evecs, rnorms, &primme);

    primme_Free(&primme);
    for (int i=0;i < primme.n * primme.numEvals;i++)
        new_us[i] = evecs[i];
    for (int i=0;i<primme.numEvals;i++){
        new_eigenvalues[i] = evals[i];
    }
}

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *v,
    lbfgsfloatval_t *g,
    const int N,
    const lbfgsfloatval_t step
    )
{
    double fx = _prob->gradientV(v,g);
    
//    cerr<<"lag obj under LFGBS = "<<fx<<endl;
    /*
    for (int i=0;i<10;i++)
	    cerr<<g[i]<<" ";
    cerr<<endl;
    exit(0);
    */
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

void non_convex_solver(){
    int N = _prob->n * _prob->rank;
    int ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N);

    /* Initialize the variables. */
    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return;
    }

    for (int i=0;i<_prob->rank;i++){
	    for (int j=0;j<_prob->n;j++){
		    int ii = i*_prob->n+j;
		    x[ii] = _prob->V[i][j];
	    }
    }
    
    lbfgs_parameter_t param;
    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.max_iterations = 1000;
    param.m = 10;
    // param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
    param.epsilon = 1e-3;//LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
    //param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */
    ret = lbfgs(N, x, &fx, evaluate, progress, NULL, &param);
 
    for (int i=0;i<_prob->rank;i++){
	    for (int j=0;j<_prob->n;j++){
		    int ii = i*_prob->n+j;
		    _prob->V[i][j] = x[ii];
	    }
    }
    /* Report the result. */
    printf("L-BFGS optimization terminated with status code = %d\n", ret);
    //exit(0); 
    lbfgs_free(x);	
}

void runHybrid(Problem* prob, Param param){
    set_prob(prob);
    double eta = param.eta;
    prob->eta = eta; 
    double old_eta = eta;
    double epsilon = 1e-3;
    int m = prob->m;
    int n = prob->n;
    int inner_iter_max = param.inner_iter;
    int outer_iter_max = param.outer_iter;
    int yt_iter_max = param.yt_iter;
    vector<double> theta;
    vector<double*>& V = prob->V;

    int& rank = prob->rank;
    int rank_capacity = 0;
    
    double obj, dobj, pinf, dinf;
    int new_k = 2;
    int max_rank = 33;//sqrt(n);
    double* new_u = new double[n];
    for (int i=0;i<n;i++)
	    new_u[i] = 0.0;
    double* new_us = new double[n*new_k];
    for (int i=0;i<n*new_k;i++)
	    new_us[i]=0.0;
    double* new_eigenvalues = new double[new_k];
    for (int i=0;i<new_k;i++)
	    new_eigenvalues[i]=0.0;

    clock_t t_start = clock();
    double t_total = 0.0;

    prob->update_res_a();
    // Initial X is zero 
    for (int yt_iter = 0;yt_iter<yt_iter_max;yt_iter++){
	    int real_new_k=0;
	    // obtain new greedy coordinates (rank 1 matrices)
	    clock_t t1 = clock();

	    if (rank<max_rank){
	    neg_grad_largest_ev(epsilon,new_k,new_us,new_eigenvalues,1); //largest algebraic eigenvector of the negative gradient
	    // push new coordinates into V
	    for (int j = 0;j < new_k;j++){
		    double eigenvalue = new_eigenvalues[j];
		    new_u = new_us + j*n;
		    if ( eigenvalue > 1e-8){ // will probably cause some problem...
			    if (rank_capacity == rank){
				    double* new_v = new double[n];
				    for (int ii=0;ii<n;ii++)
					    new_v[ii] = new_u[ii];
				    rank++;
				    rank_capacity++;
				    theta.push_back(1.0);
				    V.push_back(new_v);
			    }
			    else {
				    rank++;
				    for (int ii = 0;ii<n;ii++)
					    V[rank-1][ii] = new_u[ii];
			    }
			    real_new_k++;
		    }
	    }
	    }
	    // form theta
	    for (int j=0;j<rank;j++)
		    theta[j] = 1.0;

	    // Coordinate Descent for subproblem VSV'
	    // V is probably not orthonormal.
	    prob->solveSubProblemDiag(theta,inner_iter_max); 

	    // Form V0=V*diag(theta)
	    for (int j=0;j<rank;j++)
		    for (int ii=0;ii<n;ii++)
			    V[j][ii] = V[j][ii] * sqrt(theta[j]);

	    // peform non-convex step

	    non_convex_solver();
	    prob->update_res_a();
	    // calculate infeasibilities
	    prob->infeasibilities(pinf,dinf,obj,dobj);
//	    double duality_gap = dobj-obj;

    	    // update y
	    prob->update_y();
	    
	    t_total += ((double)(clock() - t_start))/CLOCKS_PER_SEC;

	    if( yt_iter % 1==0) 
		    cerr<<"yt iter="<<yt_iter<<", rank="<<rank<<", eta="<<eta<<", obj="<<setprecision(10)<<obj<<", dobj="<<dobj<<", pinf="<< pinf <<", dinf="<<dinf<<", time="<<t_total<<endl;
	    if (pinf < 1e-3 && dinf < 1e-3)
		    break;

//	    if (max(dinf,pinf)<1e-3)
//		    break;
    }
}

int main(int argc, char** argv){
    char* train_file;
    char* model_file = new char[FNAME_LENGTH];
    parse_command_line(argc, argv, train_file, model_file);

    srand(time(NULL));
    
    Problem* prob;
    switch(param.problem_type){
        case 1:
            cerr<<"Max Cut"<<endl<<endl;
            prob = new MaxCutProblem(train_file);
            break;
	case 2:
            cerr<<"Nearest Correlation"<<endl<<endl;
            prob = new NCMProblem(train_file);
            break;
       /* 
               
        case 3:
            cerr<<"Linear Programming"<<endl<<endl;
            prob = new LPProblem(train_file);
            break;
         */   
    }
    cerr << "dimensionality=" << prob->n <<endl;
    cerr << "number of constraints=" << prob->m <<endl;
    
    runHybrid(prob,param);

}
