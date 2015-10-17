#include "maxcut.h"
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

double* prob_a;
double prob_eta;

static Problem *_prob;

void set_prob(Problem *prob) {_prob=prob;}

bool sort_pair (const pair<int,double>& a, const pair<int,double>& b) { return (a.first < b.first); }

MaxCutProblem::MaxCutProblem(char* data_file){
    char* _line = new char[MAX_LINE];
    vector<string> tokens;

    ifstream fin(data_file);
    if (fin.fail()){
		cerr<< "can't open data file."<<endl;
		exit(0);
    }
    n = 0;
    int node_id_min = numeric_limits<int>::max();
    int node_id_max = numeric_limits<int>::min();
    while( !fin.eof()) {
        fin.getline(_line,MAX_LINE);
        string line(_line);
        split(line," ",tokens);
        if (tokens.size()<3)
            break;
        for (int edge=0;edge<2;edge++) {
            int node_id = atoi(tokens[edge].c_str());
            if (node_id < node_id_min)
                node_id_min = node_id;
            else if (node_id > node_id_max)
                node_id_max = node_id;
        }
    }
    cerr<<node_id_max<<" node id "<<node_id_min<<endl;
    n = node_id_max - node_id_min + 1;
    fin.clear();
    fin.seekg (0,fin.beg);
    C.resize(n);
    for (int i=0;i<n;i++)
        C[i] = new SparseVec();
    while( !fin.eof()) {
        fin.getline(_line,MAX_LINE);
        string line(_line);
        split(line," ",tokens);
        if (tokens.size()<3)
            break;
        int node_id0 = atoi(tokens[0].c_str());
        int node_id1 = atoi(tokens[1].c_str());
        int edge_value = atof(tokens[2].c_str());
        C[node_id0 - node_id_min]->push_back(make_pair(node_id1-node_id_min,-edge_value));
        C[node_id1 - node_id_min]->push_back(make_pair(node_id0-node_id_min,-edge_value)); // assume C is symmetric. only one direction is stored in the file.
        if (node_id0 == node_id1){
            cerr<<"should not contain self-self edge in the file"<<endl;
            cerr<< node_id0<<endl;
        }
    }
    int nnz = 0;
    for (int i=0;i<n;i++) {
        nnz += C[i]->size();
        C[i]->push_back(make_pair(i,C[i]->size()));
        sort(C[i]->begin(), C[i]->end(), sort_pair);
        for (int j=0;j<C[i]->size();j++){
            if ( j<C[i]->size()-1 && C[i]->at(j).first == C[i]->at(j+1).first ){
                cerr<<"input file contains symmetric edges (both directions)"<<endl;
                exit(0);
            }   
        }
    }
    for (int i=0;i<C.size();i++){
        for (SparseVec::iterator vit = C[i]->begin();vit != C[i]->end(); vit++){
            vit->second = - vit->second;
        }
    }

    cerr<< nnz <<endl;
    // Form A
    m = n;
    A = new SparseMat2[m];
    for (int k=0;k<m;k++){
        SparseVec2 aa;
        aa[k] = 1.0;
        A[k][k] = aa;
    }
   
    // Form b
    b = new double[m];
    for (int i=0;i<m;i++)
        b[i] = 1.0;
    
    prob_a = new double[m];
}

MaxCutProblem::~MaxCutProblem(){
    delete b;
    delete A;
    C.clear();
}

void gradVecProd(void* x, void* y, int* blockSize, primme_params* primme){
    double* xv = (double*)x;        
    double* yv = (double*)y;
    for (int i=0;i<_prob->n;i++){
        yv[i] = 0.0;
    }
    for (int k=0;k<_prob->m;k++){
        SparseMat2& Ak = _prob->A[k];
        for (SparseMat2::iterator it = Ak.begin(); it != Ak.end(); it++){
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

double MaxCutProblem::neg_grad_largest_ev(double* a,double eta, double* new_u){
    for (int i=0;i<m;i++)
        prob_a[i] = a[i];
    prob_eta = eta;
    double *evals, *evecs, *rnorms;

    /* ----------------------------- */
    /* Initialize defaults in primme */
    /* ----------------------------- */
    primme_params primme;
    primme_preset_method method;
    method = DYNAMIC;
    primme_initialize(&primme);

    /* ---------------------------------- */
    /* provide at least following inputs  */
    /* ---------------------------------- */
    primme.n = n;
    primme.eps = 1e-5;
    primme.numEvals = 1;
    primme.printLevel = 1;
    primme.matrixMatvec = gradVecProd;
    primme_set_method(method, &primme);
    primme.target = primme_largest;

    /* Allocate space for converged Ritz values and residual norms */
    evals = (double *)primme_calloc(primme.numEvals, sizeof(double), "evals");
    evecs = (double *)primme_calloc(
            primme.n*primme.numEvals,sizeof(double), "evecs");
    rnorms = (double *)primme_calloc(primme.numEvals, sizeof(double), "rnorms");

    /* ------------- */
    /*  Call primme  */
    /* ------------- */

    dprimme(evals, evecs, rnorms, &primme);

    primme_Free(&primme);
    for (int i=0;i<n;i++)
        new_u[i] = evecs[i];
    return evals[0];
}

void MaxCutProblem::uAu(double* new_u,double* new_uAu){
    for (int k=0;k<m;k++){
        double uAuk = 0.0;
        SparseMat2& Ak = A[k];
        for (SparseMat2::iterator it = Ak.begin(); it != Ak.end(); it++){
            double tmp = 0.0;
            for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
                tmp += vit->second * new_u[vit->first];
            }
            uAuk += tmp * new_u[it->first];
        }
        new_uAu[k] = uAuk;
    }   
}

double MaxCutProblem::uCu(double* new_u){
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

