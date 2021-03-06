#include "nearest_cor.h"
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

NCMProblem::NCMProblem(char* data_file){
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
	//double edge_value = 0;        
	if (node_id0 == node_id1){
		C[node_id0 - node_id_min]->push_back(make_pair(node_id1-node_id_min,edge_value));
	}
	else{
		C[node_id0 - node_id_min]->push_back(make_pair(node_id1-node_id_min,edge_value));
		C[node_id1 - node_id_min]->push_back(make_pair(node_id0-node_id_min,edge_value)); // assume C is symmetric. only one direction is stored in the file.
	}
    }
    int nnz = 0;
    for (int i=0;i<n;i++) {
	    sort(C[i]->begin(), C[i]->end(), sort_pair);
	    for (int j=0;j<C[i]->size()-1;j++){
		    if ( j<C[i]->size()-1 && C[i]->at(j).first == C[i]->at(j+1).first ){
			    cerr<<"input file contains symmetric edges (both directions)"<<endl;
			    exit(0);
		    }   
	    }
    }

    cerr<< "number of edges in problem: "<<nnz <<endl;
    
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

void NCMProblem::gradientVecProd(double* xv,//input vector
		double* yv){
	// gradient of lagrangian = 2(VV'-C)+eta A^T(A(VV') - b + y/eta);
	for (int i=0;i<n;i++){
		yv[i] = 0.0;
	}
	// A^T(A(VV')-b+y/eta)x
	for (int k=0;k<m;k++){
		SparseMat2* Ak = A[k];
		for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
			double tmp = 0.0;
			for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
				tmp += vit->second * xv[vit->first];
			}
			yv[it->first] += res_a[k] * tmp;
		}
	}

	// 2Cx
	for (int i=0;i<n;i++){
		yv[i] = -eta * yv[i];
		double tmp = 0.0;
		for (SparseVec::iterator vit = C[i]->begin();vit != C[i]->end(); vit++){
			tmp += vit->second * xv[vit->first];
		}
		yv[i] += 2*tmp;
	}
	// V'x
	double* VTx = new double[rank];
	for (int j=0;j<rank;j++){
		double tmp=0.0;
		for (int i=0;i<n;i++)
			tmp += V[j][i]*xv[i];
		VTx[j] = tmp;
	}
	// 2VV'x
	for (int j=0;j<rank;j++){
		for (int i=0;i<n;i++)
			yv[i] -= 2*V[j][i]*VTx[j];
	}
	delete[] VTx;
	
	/*for (int i=0;i<10;i++)
		cerr<<xv[i]<<" ";
	cerr<<endl;
	for (int i=0;i<10;i++)
		cerr<<yv[i]<<" ";
	cerr<<endl;
	exit(0);
	*/
}; 

void NCMProblem::update_res_a(){
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
//  grad = 4(VV'-C)V + 2etaA^T(A(VV')-b+y/eta)V
double NCMProblem::gradientV(const double* v, double *g){
	// copy v to V
	for (int i=0;i<rank;i++){
		const double* new_v = v + n*i;
		for (int j=0;j<n;j++){
			V[i][j] = new_v[j];
		}
	}
	update_res_a();
	double fx = 0.0;
    for (int ii=0;ii<n*rank;ii++)
	    g[ii] = 0.0;
    for (int k=0;k<m;k++){
	    fx += res_a[k]*res_a[k];
    }
    fx *= (eta/2);
    // 2etaA^T(A(VV')-b+y/eta)V
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
    // C part, fx = -2tr(V'CV), grad = -2CV;
    for (int i=0;i<rank;i++){
	    double* new_v = V[i];
	    double* gi = g + n*i;
	    for (int j=0;j<C.size();j++){
		    if (C[j]->size()>0){
		    double tmp = 0.0;
		    for (SparseVec::iterator vit = C[j]->begin();vit !=C[j]->end(); vit++){
			    tmp += vit->second * new_v[vit->first];
		    }
		    fx -= 2*tmp * new_v[j];
		    gi[j] -= 4*tmp;
		    }
	    }
    }
    // V'V
    
    vector<double*> VTV;
    for (int i=0;i<rank;i++){
	    double* empv = new double[rank];
    	    for (int j=0;j<rank;j++){
		    double tmp = 0.0;
		    for (int k=0;k<n;k++)
			    tmp += V[i][k]*V[j][k];
		    empv[j] = tmp;
		    fx += tmp*tmp;
	    }
	    VTV.push_back(empv);
    }
    // VV'V part, fx = \|V'V\|_F^2, grad = 4VV'V
    for (int i=0;i<rank;i++){
	    double* gi = g+n*i;
    	for (int j=0;j<n;j++){
		double tmp = 0.0;
		for (int k=0;k<rank;k++)
			tmp += V[k][j]*VTV[i][k];
		gi[j] += 4*tmp;
	}
    }
    return fx;
};

void NCMProblem::solveSubProblemDiag(vector<double>& theta,int inner_iter_max){
	// form VAV
	vector<double*> VAV;
	for (int i=0;i<rank;i++){
		double* new_v = V[i];
		double* empv = new double[m];
		//VAV2[i] = 0.0;
		for (int k=0;k<m;k++){
			empv[k]=0;
			SparseMat2* Ak = A[k];
			for (SparseMat2::iterator it = Ak->begin(); it != Ak->end(); it++){
				double tmp = 0.0;
				for (SparseVec2::iterator vit = it->second.begin();vit != it->second.end(); vit++){
					tmp += vit->second * new_v[vit->first];
				}
				empv[k] += tmp*new_v[it->first];	    
			}
		//	VAV2[i] += empv[k] * empv[k];
		}
		VAV.push_back(empv);
	}

	// for VAV2 
	vector<double*> VAV2;
	    
	for (int i=0;i<rank;i++){
		double* empv = new double[rank];
		for (int j=0;j<rank;j++){
			double tmp = 0.0;
			for (int k=0;k<m;k++)
				tmp += VAV[i][k]*V[j][k];
			empv[j] = tmp;
		}
		VAV2.push_back(empv);
	}

	// form eta_VAV_by
	double* VAV_by = new double[rank];
	for (int i = 0;i<rank;i++){	
		VAV_by[i] = 0.0;
		for (int k=0;k<m;k++){
			VAV_by[i] += VAV[i][k]*(eta*b[k]-y[k]);
		}
	}
	
	// form VTV2 = (V'V).^2
	vector<double*> VTV2;
	for (int i=0;i<rank;i++){
		double* empv = new double[rank];
		for (int j=0;j<rank;j++){
			double tmp = 0.0;
			for (int k=0;k<n;k++)
				tmp += V[i][k]*V[j][k];
			empv[j] = tmp*tmp;
		}
		VTV2.push_back(empv);
	}

	// form VTV2theta = VTV2*theta
/*		double* VTV2theta = new double[rank];
	for (int k=0;k<rank;k++){
		VAVtheta_by[k] = 0.0;
		for (int i=0;i<rank;i++)
			VAVtheta_by[k] += VTV2[k][i]*theta[k];
	}
*/
	// form vCv
	double* vCv = new double[rank]; 
	for (int i=0;i<rank;i++){
		vCv[i] = 0.0;
		double* new_v = V[i];
		for (int j=0;j<C.size();j++){
			if (C[j]->size()>0){
				double tmp = 0.0;
				for (SparseVec::iterator vit = C[j]->begin();vit !=C[j]->end(); vit++){
					tmp += vit->second * new_v[vit->first];
				}
				vCv[i] += tmp * new_v[j];
			}
		}
	}

	// fully corrective update for the coefficients of coordinates, theta
            vector<int> innerAct;
            for (int i=rank-1;i>=0;i--)
                innerAct.push_back(i);
	    for (int inner_iter=0;inner_iter<inner_iter_max;inner_iter++){
		    random_shuffle(innerAct.begin(),innerAct.end());
                for (int k = 0;k<rank;k++){
                    int j = innerAct[k];
            
		    double quad_coeff = VTV2[j][j] + (eta/2) * VAV2[j][j];
		    double aug_linear_coeff = 0;
		    for (int l=0;l<rank;l++){
			    aug_linear_coeff += VAV2[j][l]*theta[l];// VAV[j][l]*(VAVtheta_by[l] - VAV[j][l]*theta[j]);
		    }

		    aug_linear_coeff *= eta;
		    aug_linear_coeff -= VAV_by[j];

		    double linear_coeff = 0.0;
		    for (int l=0;l<rank;l++)
			    linear_coeff += VTV2[j][l]*theta[l];
		    linear_coeff -= VTV2[j][j]*theta[j];
		    linear_coeff *= 2;
		    linear_coeff -= 2 *vCv[j];
		    linear_coeff += aug_linear_coeff;

	    	    double delta_theta = -linear_coeff/(2*quad_coeff) - theta[j];
                    if (delta_theta > -theta[j]){
                        theta[j] += delta_theta;
                    }
                    else {
                        theta[j] = 0;
                    }
                }
            }
            // remove the coordinates with zero (<1e-6) coefficient from active set
            for (int j=0;j<rank;j++){
                if (fabs(theta[j]) < 1e-6) {
                    theta[j] = theta[rank-1];
                    theta[rank-1] = 0.0;
		    V[j] = V[rank-1];
		    rank--;
                    j--;
                }
            }

	delete[] vCv;
};

void NCMProblem::update_y(){
	for (int i=0;i<m;i++)
		y[i] = eta*res_a[i];
}; 

void NCMProblem::infeasibilities(double& pinf, double& dinf, double& obj, double& dobj){
	obj = primal_obj();
	pinf = primal_inf();
};

double NCMProblem::primal_obj(){
	double obj = 0;
	// C part, fx = -2tr(V'CV), grad = -2CV;
	for (int i=0;i<rank;i++){
		double* new_v = V[i];
		for (int j=0;j<C.size();j++){
			double tmp = 0.0;
			for (SparseVec::iterator vit = C[j]->begin();vit !=C[j]->end(); vit++){
				tmp += vit->second * new_v[vit->first];
			}
			obj -= 2*tmp * new_v[j];
		}
	}
	// V'V
	for (int i=0;i<rank;i++){
		for (int j=0;j<rank;j++){
			double tmp = 0.0;
			for (int k=0;k<n;k++)
				tmp += V[i][k]*V[j][k];
			obj += tmp*tmp;
		}
	}
	return obj;

};

double NCMProblem::primal_inf(){
	double pinf=0.0;
	for (int i=0;i<m;i++){
		pinf = max(fabs(res_a[i] - y[i]/eta),pinf);
	}
	return pinf;
};

NCMProblem::~NCMProblem(){
	delete b;
	A.clear();
	C.clear();
}

