#include "problem.h"
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

double* prob_a;
double prob_eta;

static Problem *_prob;

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


double Problem::neg_grad_largest_ev(double* a,double eta, double epsilon,double* new_u){
    
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
    method = JDQMR_ETol;
    primme_initialize(&primme);

    /* ---------------------------------- */
    /* provide at least following inputs  */
    /* ---------------------------------- */
    primme.n = n;
    primme.eps = epsilon;
    primme.numEvals = 1;
    primme.printLevel = 2;
    primme.matrixMatvec = gradVecProd;
    primme.target = primme_largest;
    primme_set_method(method, &primme);

    /* Allocate space for converged Ritz values and residual norms */
    evals = (double *)primme_calloc(primme.numEvals, sizeof(double), "evals");
    evecs = (double *)primme_calloc(
            primme.n*primme.numEvals,sizeof(double), "evecs");
    rnorms = (double *)primme_calloc(primme.numEvals, sizeof(double), "rnorms");
//    for(int i=0;i<n;i++)
//        evecs[i] = new_u[i];
    /* ------------- */
    /*  Call primme  */
    /* ------------- */
    //double start2 = omp_get_wtime();
    dprimme(evals, evecs, rnorms, &primme);
    //cerr << "eig solve time=" << omp_get_wtime()-start2 << endl;
    
    primme_Free(&primme);
    for (int i=0;i<n;i++)
        new_u[i] = evecs[i];
    
    //cerr << "whole time=" << omp_get_wtime()-start << endl;

    return evals[0];
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

