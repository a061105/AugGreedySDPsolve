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
            cerr<< node_id0<<endl;
        }
    }
    int nnz = 0;
    for (int i=0;i<n;i++) {
        nnz += C[i]->size();
        sort(C[i]->begin(), C[i]->end(), sort_pair);
        for (int j=0;j<C[i]->size();j++){
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
    
    allocate_prob_a(m);
}


MaxCutProblem::~MaxCutProblem(){
    delete b;
    A.clear();
    C.clear();
}

