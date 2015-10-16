#include<math.h>
#include<vector>
#include<cstring>
#include<stdlib.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<deque>
#include <ctime>
#include <iomanip>  
#include "problem.h"
#include "maxcut.h"

using namespace std;

class Param{
	public:
	Param(){
		problem_type = 1;
		solver = 0;
		info_file = NULL;
		max_iter = 1000;
        lambda = 1.0;
        epsilon = 1e-3;
	}
	int problem_type;
	int solver;
	char* info_file;
	int max_iter;
    double lambda;
    double epsilon; //termination criterion
};

Param param;
void exit_with_help(){

    cerr << "Usage: ./sdp_omp (options) [train_data] (model)" << endl;
    cerr << "options:" << endl;
    cerr << "-p problem_type: (default 1)" << endl;
    cerr << "	1 -- max cut" << endl;
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
            case 'i': param.info_file = argv[i];
                      break;
            case 'm': param.max_iter = atoi(argv[i]);
                      break;
            case 'l': param.lambda = atof(argv[i]);
                      break;
            case 'e': param.epsilon = atof(argv[i]);
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

void runOMP(Problem* prob){
    set_prob(prob);
    double eta = 1e-3;
    int m = prob->m;
    int n = prob->n;
    double* b = prob->b;
    int inner_iter_max = 30;
    int outer_iter_max = 10;
    int yt_iter_max = 1000;
    vector<double> theta;
    vector<double*> B;
    vector<double> c;

    int num_rank1 = 0;
    int num_rank1_capacity = 0;
    double* a = new double[m];
    double* y = new double[m];
    for (int i=0;i<m;i++){
        a[i] = -(b[i] - y[i]/eta);
        y[i] = 0.0;
    }
    double* new_u = new double[n];
    for (int yt_iter = 0;yt_iter<yt_iter_max;yt_iter++){
        num_rank1 = 0;
        for (int i=0;i<m;i++)
            a[i] = -(b[i] - y[i]/eta);
        
        for (int outer_iter= 0;outer_iter<outer_iter_max;outer_iter++){
            double eigenvalue = prob->neg_grad_largest_ev(a,eta,new_u); //largest algebraic eigenvector of the negative gradient
            if (eigenvalue > 1e-6){
                double new_c = prob->uCu(new_u);
                if (num_rank1_capacity == num_rank1){
                    num_rank1++;
                    num_rank1_capacity++;
                    double* new_uAu = new double[m];
                    prob->uAu(new_u,new_uAu);
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
                for (int i=0;i<m;i++)
                    a[i] += B[num_rank1-1][i];
            }
            vector<int> innerAct;
            for (int i=num_rank1-1;i>=0;i--)
                innerAct.push_back(i);
            for (int inner_iter=0;inner_iter<inner_iter_max;inner_iter++){
                random_shuffle(innerAct.begin()+1,innerAct.end());
                for (int k = 0;k<num_rank1;k++){
                    int j = innerAct[k];
                    double delta_theta = -(eta*dot(a,B[j],m)+c[j])/(eta*l2_norm(B[j],m)*2.0);
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
            double obj = dot(c,theta) + eta * l2_norm(a,m);
            for (int j=0;j<num_rank1;j++){
                if (fabs(theta[j]) < 1e-12) {
                    theta[j] = theta[num_rank1-1];
                    B[j] = B[num_rank1-1];
                    c[j] = c[num_rank1-1];
                }

            }
            cerr<<"outer iter="<<outer_iter<<", obj="<<obj<<endl;
        }
        for (int i=0;i<m;i++)
            y[i] = eta*a[i];

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
    }
    cerr << "dimensionality=" << prob->n <<endl;
    runOMP(prob);

}
