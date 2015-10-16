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
#include "seq_label.h"
#include "taxonomy.h"

#include "optimizer.h"
#include "proxQN.h"

using namespace std;

Param param;

void exit_with_help(){

    cerr << "Usage: ./train (options) [train_data] (model)" << endl;
    cerr << "options:" << endl;
    cerr << "-p problem_type: (default 1)" << endl;
    cerr << "	1 -- Sequence labeling (info file = feature template)" << endl;
    cerr << "	2 -- Hierarchical classification (info file = taxonomy tree)" << endl;
    cerr << "-i info: (additional info specified in a file)" << endl;
    cerr << "-m max_iter: maximum_outer_iteration (default 100)" << endl;
    cerr << "-l lambda: regularization coefficient (default 1.0)"<<endl;	
    cerr << "-e epsilon: stop criterion (default 1e-6)"<<endl;
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

void writeModel( const char* fname, double* w, int d, int raw_d){
    ofstream fout(fname);
    fout << "number of raw features:" << raw_d << endl;

    for(int i=0;i<d;i++){
        if (w[i] != 0.0)
        fout << i << ":" << w[i]<<" ";
    }
    fout<<endl;
    fout.close();
}

int main(int argc, char** argv){
    char* train_file;
    char* model_file = new char[FNAME_LENGTH];
    parse_command_line(argc, argv, train_file, model_file);

    srand(time(NULL));
    
    optimizer* opt = new proxQN();
    
    Problem* prob;
    switch(param.problem_type){
        case 0:
            cerr<<"multiclass classification not ready"<<endl;
            exit(0);
        case 1:
            cerr<<"Sequence labeling problem."<<endl<<endl;
            prob = new SeqLabelProblem(train_file);
            break;
        case 2:
            cerr<<"Hierarchical classification problem."<<endl<<endl;
            prob = new TaxonomyProblem(train_file);     
            break;
        case 3:
            cerr<<"sequence alignment not ready"<<endl;
            exit(0);
        case 4:
            cerr<<"sequence parsing not ready"<<endl;
            exit(0);
    }
    cerr << "number of weights: " << prob->d << endl;
    cerr<<"number of samples: "<< prob->N <<endl<<endl;	
    opt->lambda = param.lambda;
    opt->max_iter = param.max_iter;
    opt->epsilon = param.epsilon;

    opt->minimize(prob);

    writeModel(model_file, prob->w, prob->d, prob->raw_d);

}
