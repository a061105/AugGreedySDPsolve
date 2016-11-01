#include "problem.h"
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
        double tmp = 0.0;
        for (SparseVec::iterator vit = _prob->C[i]->begin();vit != _prob->C[i]->end(); vit++){
            tmp += vit->second * xv[vit->first];
        }
        yv[i] -= tmp;
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
    for (int i=0;i<C.size();i++){
        double tmp = 0.0;
        for (SparseVec::iterator vit = C[i]->begin();vit != C[i]->end(); vit++){
            tmp += vit->second * new_u[vit->first];
        }
        uCuv += new_u[i] * tmp;
    }
    return uCuv;
}

