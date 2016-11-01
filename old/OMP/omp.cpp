#include<math.h>
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
		alpha = 0.1;
	}
	int problem_type;
	int inner_iter;
	int outer_iter;
	int yt_iter;
	double eta;
	double alpha;
};

Param param;
void exit_with_help(){

    cerr << "Usage: ./sdp_omp (options) [train_data] (model)" << endl;
    cerr << "options:" << endl;
    cerr << "-p problem_type: (default 1)" << endl;
    cerr << "	1 -- max cut" << endl;
    cerr << "	2 -- linear programming" << endl;
    cerr << "-t step size (default 0.1)" << endl;
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
            case 't': param.alpha = atof(argv[i]);
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

void runOMP(Problem* prob, Param param){
    set_prob(prob);
    double eta = param.eta;
    double old_eta = eta;
    double alpha = param.alpha;
    double epsilon = 1e-1;
    int m = prob->m;
    int n = prob->n;
    double* b = prob->b;
    int inner_iter_max = param.inner_iter;
    int outer_iter_max = param.outer_iter;
    int yt_iter_max = param.yt_iter;
    vector<double> theta;
    vector<double*> B;
    vector<double> c;

    int num_rank1 = 0;
    int num_rank1_capacity = 0;
    double* a = new double[m];
    double* y = new double[m];
    double* old_y = new double[m];
    
    for (int i=0;i<m;i++){
        y[i] = 0.0;
        old_y[i] = 0.0;
        a[i] = -(b[i] - y[i]/eta);
    }
    double obj, dobj, pinf, dinf;
    int new_k = 4;
    double* new_u = new double[n];
    for (int i=0;i<n;i++)
        new_u[i] = 0.0;
    double* new_us = new double[n*new_k];
    for (int i=0;i<n*new_k;i++)
        new_us[i]=0.0;
    double* new_eigenvalues = new double[new_k];
    for (int i=0;i<new_k;i++)
        new_eigenvalues[i]=0.0;
    double* pinfea = new double[m];
    double b_inf = inf_norm(b,m);
/*
    // testing primme performance, remember matvec function is also changed in this mode...
    new_k = 5;
    double pre = 1e-1;
    double err;
    double aa,bb,bb_plus;
    int repNum = 100;
    double* ground_us = new double[n*new_k];
    for (int i=0;i<n*new_k;i++)
        ground_us[i]=0.0;

    prob->neg_grad_largest_ev(a,0.0,1e-12,new_k,ground_us,new_eigenvalues,1); 

    for (int exp = 0;exp < 10; exp++){
	    clock_t t1 = clock();
	    err = 0.0;
	    for (int rep = 0;rep<repNum;rep++){
		    bb = 0.0;
		    bb_plus = 0.0;
		    prob->neg_grad_largest_ev(a,0.0,pre,new_k,new_us,new_eigenvalues,1); 
		    for (int i=0;i<n;i++) {
		    	    aa = new_us[i] - ground_us[i];
			    bb += aa*aa;
		    }
		    for (int i=0;i<n;i++) {
			    aa = new_us[i] + ground_us[i];
			    bb_plus += aa*aa;
		    }
		   
		    err += sqrt(min(bb,bb_plus));
	    }
	    cerr<<"precision="<<pre<<", time="<<(double)(clock()-t1)/CLOCKS_PER_SEC<<", error="<<err/repNum<<endl;
	    pre /= 2.0;
    }
    exit(0);
   */ 
    prob->neg_grad_largest_ev(a,0.0,1e-8,new_k,new_us,new_eigenvalues,0); 
    double C_spectral = fabs(new_eigenvalues[0]);
    prob->neg_grad_largest_ev(a,0.0,1e-8,new_k,new_us,new_eigenvalues,1);
    C_spectral = max(fabs(new_eigenvalues[0]),C_spectral);
    
    clock_t tstart = clock();
    bool apply_non_negative_constraint = true;
    int phase = 1;
    double t_inner=0.0;
    for (int yt_iter = 0;yt_iter<yt_iter_max;yt_iter++){
        //num_rank1 = 0;
        for (int i=0;i<m;i++){
            //a[i] = -(b[i] - y[i]/eta);
            a[i] = a[i] + y[i]/eta - old_y[i]/old_eta;
            old_y[i] = y[i];
        }
        for (int outer_iter= 0;outer_iter<outer_iter_max;outer_iter++){
            int real_new_k=0;
            // obtain new greedy coordinates (rank 1 matrices)
            double t_eig = 0.0;
            clock_t t1 = clock();
            prob->neg_grad_largest_ev(a,eta,epsilon,new_k,new_us,new_eigenvalues,1); //largest algebraic eigenvector of the negative gradient
            t_eig += ((double)(clock() - t1))/CLOCKS_PER_SEC;
            // push new coordinates into active set if the corresponding eigenvalue is greater than 1e-8
            for (int j = 0;j < new_k;j++){
                double eigenvalue = new_eigenvalues[j];
                new_u = new_us + j*n;
                if ( eigenvalue > 1e-8 || outer_iter==0){
                    double new_c = prob->uCu(new_u);
                    if (num_rank1_capacity == num_rank1){
                        num_rank1++;
                        num_rank1_capacity++;
                        double* new_uAu = new double[m];
                        prob->uAu(new_u, new_uAu);
                        theta.push_back(1.0);
                        B.push_back(new_uAu);
                        c.push_back(new_c);
                    }
                    else {
                        num_rank1++;
                        prob->uAu(new_u,B[num_rank1-1]);
                        theta[num_rank1-1] = 1.0;
                        c[num_rank1-1] = new_c;
                    }
                    for (int i=0;i<m;i++){
                        a[i] += theta[num_rank1-1]*B[num_rank1-1][i];
                    }
                    real_new_k++;
                }
            }

            // fully corrective update for the coefficients of coordinates, theta
            vector<int> innerAct;
            for (int i=num_rank1-1;i>=0;i--)
                innerAct.push_back(i);
	    clock_t t_inner_start = clock();
	    for (int inner_iter=0;inner_iter<inner_iter_max;inner_iter++){
                if (inner_iter==0 && innerAct.size()>1){
                    random_shuffle(innerAct.begin()+real_new_k,innerAct.end()); // always update the newly added coordinates 
                }
                else {
                    random_shuffle(innerAct.begin(),innerAct.end());
                }
                for (int k = 0;k<num_rank1;k++){
                    int j = innerAct[k];
                    double delta_theta = -(eta*dot(a,B[j],m)+c[j])/(eta*l2_norm_square(B[j],m));
                    if (delta_theta > -theta[j]){
                        theta[j] += delta_theta;
                        for (int i=0;i<m;i++)
                            a[i] += delta_theta*B[j][i];
                    }
                    else {
                        for (int i=0;i<m;i++)
                            a[i] -= theta[j]*B[j][i];
                        theta[j] = 0;
                    }
                }
            }
	    t_inner += (double)(clock()-t_inner_start)/CLOCKS_PER_SEC;
            // remove the coordinates with zero (<1e-6) coefficient from active set
            for (int j=0;j<num_rank1;j++){
                if (fabs(theta[j]) < 1e-6) {
                    theta[j] = theta[num_rank1-1];
                    double* tmpB = B[j];
                    B[j] = B[num_rank1-1];
                    B[num_rank1-1] = tmpB;
                    c[j] = c[num_rank1-1];
                    theta[num_rank1-1] = 0.0;
                    c[num_rank1-1] = 0.0;
                    num_rank1--;
                    j--;
                }
            }
            obj = dot(c,theta);// + eta/2.0 * l2_norm_square(a,m);
        }
        // calculate infeasibilities
        double old_pinf = pinf;
	for (int j=0;j<m;j++)
            pinfea[j] = a[j] - y[j]/eta;
        // pinf = inf_norm(pinfea,m);
        pinf = inf_norm(pinfea,m)/(1.0+b_inf);
        
        for (int i=0;i<m;i++)
		y[i] += alpha*eta*(a[i]-y[i]/eta);

	if( yt_iter % 20==0){ 
		prob->neg_grad_largest_ev(y,1.0,1e-8,1,new_u,new_eigenvalues,1);
		if (new_eigenvalues[0]>0)
			dinf = new_eigenvalues[0]/(C_spectral+1);
		else
			dinf = 0.0;
	}
        dobj = -dot(y,b,m);    
        double duality_gap = fabs(dobj-obj)/(1+fabs(dobj)+fabs(obj));
        clock_t t_yt = clock();
	if( yt_iter % 20==0) 
        	cerr<<"yt iter="<<yt_iter<<", t="<<((double)(t_yt - tstart))/CLOCKS_PER_SEC<<", t_inner="<<t_inner<<", obj="<<setprecision(10)<<obj<<", dobj="<<dobj<<", pinf="<< pinf <<", dinf="<<dinf<<", Gap="<<duality_gap<<endl;
       
	old_eta = eta;
	// update eta
 	eta *=1.001;
/*
	if( yt_iter % 20==0){ 
	if ( pinf > 10 * dinf){
		eta *= 2.0;
	}
	else if (dinf > 10 * pinf){
		eta /=2.0;
	}
	}
*/
	// change phases
        // phase=1: starting phase, when infeasiblities are large, so no need to solve eigenvalue problem very accurately. use large epsilon
        // phase=2: try to decrease infeasiblity to small values. use small epsilon
        if( phase==1 && pinf < 1.0 && yt_iter > 100 ){
            // epsilon = 1e-4;
            phase=2;
        }
        if( phase==2 && max(dinf,pinf) < 1e-4 ){
            // epsilon = 1e-6;
            //outer_iter_max = 5;
            phase = 3;
        }
	if( yt_iter % 20==0){ 
		cerr<<"phase " << phase <<", num_rank1="<<num_rank1<<", eta="<<eta<<", epsilon="<<epsilon<<endl;
		if (max(dinf,pinf)<1e-3)
			break;
	}
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
            cerr<<"Linear Programming"<<endl<<endl;
            prob = new LPProblem(train_file);
            break;
            
    }
    cerr << "dimensionality=" << prob->n <<endl;
    cerr << "number of constraints=" << prob->m <<endl;
    
    runOMP(prob,param);

}
