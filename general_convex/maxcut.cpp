#include "maxcut.h"
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

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
        double edge_value = atof(tokens[2].c_str());
        C[node_id0 - node_id_min]->push_back(make_pair(node_id1-node_id_min,edge_value));
        C[node_id1 - node_id_min]->push_back(make_pair(node_id0-node_id_min,edge_value)); // assume C is symmetric. only one direction is stored in the file.
        if (node_id0 == node_id1){
            cerr<<"should not contain self-self edge in the file"<<endl;
            cerr<< node_id0<<" " <<node_id1<<endl;
        }
    }
    int nnz = 0;
    for (int i=0;i<n;i++) {
	    double tmp = 0.0;
	    for (SparseVec::iterator vit = C[i]->begin();vit != C[i]->end(); vit++)
		    tmp += vit->second ;
	    nnz += C[i]->size();
	    C[i]->push_back(make_pair(i,-tmp));
	    sort(C[i]->begin(), C[i]->end(), sort_pair);
	    for (int j=0;j<C[i]->size()-1;j++){
		    if ( j<C[i]->size()-1 && C[i]->at(j).first == C[i]->at(j+1).first ){
			    cerr<<"input file contains symmetric edges (both directions)"<<endl;
			    exit(0);
		    }   
	    }
    }

    cerr<< "number of edges in max cut problem: "<<nnz <<endl;
    
    // Form A
    m = n;
    A.clear();
    
    for (int k=0;k<m;k++){
        SparseVec2 aa;
        aa[k] = 1.0;
        SparseMat2* Ak = new SparseMat2();
        Ak->insert(make_pair(k,aa));
        A.push_back(Ak);
    }
  
    // Form b
    b = new double[m];
    for (int i=0;i<m;i++)
        b[i] = 1.0;
   
    res_a = new double[m];
    rank = 0;
    y = new double[m];
    for (int i=0;i<m;i++)
        y[i] = 0.0;

}

void MaxCutProblem::gradientVecProd(double* xv,//input vector
		double* yv){
	for (int i=0;i<n;i++){
		yv[i] = 0.0;
	}
	for (int k=0;k<m;k++){
		SparseMat2* Ak = A[k];
		for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
			double tmp = 0.0;

			for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
				tmp += vit->second * xv[vit->first];
			}
			yv[it->first] += res_a[k]*tmp;
		}
	}
	for (int i=0;i<n;i++){
		yv[i] = -eta * yv[i];
		double tmp = 0.0;
		for (SparseVec::iterator vit = C[i]->begin();vit != C[i]->end(); vit++){
			tmp += vit->second * xv[vit->first];
		}
		yv[i] -= tmp;
	}
}; 

void MaxCutProblem::update_res_a(){
    for (int k=0;k<m;k++){
	    SparseMat2* Ak = A[k];
	    res_a[k]=0;
	    for (int i=0;i<rank;i++){
		    double* new_v = V[i];
		    for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
			    double tmp = 0.0;
			    for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
				    tmp += vit->second * new_v[vit->first];
			    }
			    res_a[k] += tmp * new_v[it->first];
		    }
	    }
	    res_a[k] = res_a[k] - b[k] + y[k]/eta;
    }
}; 
// calculate the function value (returned) and the gradient of V 
double MaxCutProblem::gradientV(const double* v, double *g){
// copy v to V
	for (int i=0;i<rank;i++){
		const double* new_v = v + n*i;
		for (int j=0;j<n;j++)
			V[i][j] = new_v[j];
	}
	update_res_a();
	double fx = 0.0;
    for (int ii=0;ii<n*rank;ii++)
	    g[ii] = 0.0;
    for (int k=0;k<m;k++){
	    fx += res_a[k]*res_a[k];
    }
    fx *= (eta/2);
    for (int k=0;k<m;k++){
    	    SparseMat2* Ak = A[k];
	    for (int i=0;i<rank;i++){
		    double* new_v = V[i];
		    double* gi = g + n*i;
		    for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
			    double tmp = 0.0;
			    for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
				    tmp += vit->second * new_v[vit->first];
			    }
			    gi[it->first] += 2 * eta * tmp * res_a[k];
		    }
	    }
    }
    // C part
    for (int i=0;i<rank;i++){
	    double* new_v = V[i];
	    double* gi = g + n*i;
	    for (int j=0;j<C.size();j++){
		    double tmp = 0.0;
		    for (SparseVec::iterator vit = C[j]->begin();vit !=C[j]->end(); vit++){
			    tmp += vit->second * new_v[vit->first];
		    }
		    fx += tmp * new_v[j];
		    gi[j] += 2*tmp;
	    }
    }

	return fx;
};

void MaxCutProblem::solveSubProblemDiag(double* theta){
	
};

void MaxCutProblem::update_y(){
	for (int i=0;i<m;i++)
		y[i] = eta*res_a[i];
}; 

void MaxCutProblem::infeasibilities(double& pinf, double& dinf, double& obj, double& dobj){
	obj = primal_obj();
	pinf = primal_inf();
};

double MaxCutProblem::primal_obj(){
	double obj = 0;
	for (int i=0;i<rank;i++){
		double* new_v = V[i];
		for (int j=0;j<C.size();j++){
			double tmp = 0.0;
			for (SparseVec::iterator vit = C[j]->begin();vit !=C[j]->end(); vit++){
				tmp += vit->second * new_v[vit->first];
			}
			obj += tmp * new_v[j];
		}
	}

	return obj;

};

double MaxCutProblem::primal_inf(){
	double pinf=0.0;
	for (int i=0;i<m;i++){
		pinf = max(fabs(res_a[i] - y[i]/eta),pinf);
	}
	return pinf;
};

MaxCutProblem::~MaxCutProblem(){
	delete b;
	A.clear();
	C.clear();
}

