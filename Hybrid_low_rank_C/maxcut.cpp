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
    vector<int> ns;
    while( !fin.eof()) {
        fin.getline(_line,MAX_LINE);
        string line(_line);
        split(line," ",tokens);
        if (tokens.size()<1)
            break;
	n = tokens.size();
	ns.push_back(n);
	double* cur_vec = new double[n];
	for (int ii=0;ii<n;ii++)
		cur_vec[ii] = atof(tokens[ii].c_str());
	VC.push_back(cur_vec);
    }
    // check for ns
    for (int ii=0;ii<ns.size();ii++)
	    if (n != ns[ii]){
		    cerr<<"row size inconsistent"<<endl;
		    exit(0);
	    }

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
    
    allocate_prob_a(m);
}


MaxCutProblem::~MaxCutProblem(){
    delete b;
    A.clear();
    VC.clear();
}

