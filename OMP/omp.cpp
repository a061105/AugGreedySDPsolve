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

void runOMP(Problem* prob, Param param){
    set_prob(prob);
    double eta = param.eta;
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
   /* 
    double y0[] = {
        -0.137135749628878,
        -0.932595463037164,
            -0.696818161489900,
            -0.066000172722062,
            -0.755463052602466,
            -0.753876188461246,
            -0.923024535546483,
            -0.711524758628472,
            -0.124270961972165,
            -0.019880133839796};
    double y0[] = {
        -9.350688759773709,
        -10.476659905165564,
        -10.106033458265545,
        -8.245951865763066,
        -10.178233242802358,
        -8.590050176232541,
        -9.524656839317776,
        -9.320225797143534,
        -9.093357607129120,
        -10.395308416191948};
        */
    for (int i=0;i<m;i++){
        y[i] = 0.0;
        a[i] = -(b[i] - y[i]/eta);
    }
    double obj;
    double* new_u = new double[n];
    double* infea = new double[m];
    for (int yt_iter = 0;yt_iter<yt_iter_max;yt_iter++){
        num_rank1 = 0;
        for (int i=0;i<m;i++)
            a[i] = -(b[i] - y[i]/eta);
        
        for (int outer_iter= 0;outer_iter<outer_iter_max;outer_iter++){
            double eigenvalue = prob->neg_grad_largest_ev(a,eta,new_u); //largest algebraic eigenvector of the negative gradient
            if (eigenvalue > 1e-12){
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
                for (int i=0;i<m;i++){
                    a[i] += theta[num_rank1-1]*B[num_rank1-1][i];
                }
            }
            vector<int> innerAct;
            for (int i=num_rank1-1;i>=0;i--)
                innerAct.push_back(i);
            for (int inner_iter=0;inner_iter<inner_iter_max;inner_iter++){
                //random_shuffle(innerAct.begin()+1,innerAct.end());
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
            for (int j=0;j<num_rank1;j++){
                if (fabs(theta[j]) < 1e-12) {
                    theta[j] = theta[num_rank1-1];
                    double* tmpB = B[j];
                    B[j] = B[num_rank1-1];
                    B[num_rank1-1] = tmpB;
                    c[j] = c[num_rank1-1];
                    theta[num_rank1-1] = 0.0;
                    c[num_rank1-1] = 0.0;
                    num_rank1--;
                }
            }
            obj = dot(c,theta) + eta/2.0 * l2_norm_square(a,m);
           // cerr<<"outer iter="<<outer_iter<<", obj="<<setprecision(10)<<obj<<endl;
        }
        cerr<<"num_rank1="<<num_rank1<<endl;
        for (int j=0;j<m;j++)
            infea[j] = a[j] - y[j]/eta;
        cerr<<"yt iter="<<yt_iter<<", obj="<<setprecision(10)<<obj<<", infeasibility="<<sqrt(l2_norm_square(infea,m))<<endl;
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
    runOMP(prob,param);

}
