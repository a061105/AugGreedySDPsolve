#include "lp.h"
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

LPProblem::LPProblem(char* data_folder){
    char* _line = new char[MAX_LINE];
    vector<string> tokens;

    strcat(data_folder,"/c");
    ifstream c_file(data_folder);
    if (c_file.fail()){
		cerr<< "can't open "<<data_folder<<endl;
		exit(0);
    }
    
    data_folder[strlen(data_folder)-1] = 'A';
    ifstream A_file(data_folder);
    if (A_file.fail()){
		cerr<< "can't open "<<data_folder<<endl;
		exit(0);
    }
    
    data_folder[strlen(data_folder)-1] = 'b';
    ifstream b_file(data_folder);
    if (b_file.fail()){
		cerr<< "can't open "<<data_folder<<endl;
		exit(0);
    }

    // Processing c file
    C.clear();
    int n_raw = 0;
    while(!c_file.eof()){
        c_file.getline(_line,MAX_LINE);
        string line(_line);
        split(line,"\t",tokens);
        if (tokens.size()==0)
            break;

        SparseVec* sv = new SparseVec();
        sv->push_back(make_pair(n_raw,atof(_line)));
        C.push_back(sv);
        n_raw++;
    }
    
    // Processing A
    A_file.getline(_line,MAX_LINE);
    
    A.clear();
    
    m = -1;
    SparseMat2* Ak;// = new SparseMat2();
    while(!A_file.eof()) {
        A_file.getline(_line,MAX_LINE);
        string line(_line);
        split(line,"\t",tokens);
        if (tokens.size()<3)
            break;
        int ii,jj,vv;
        ii = atoi(tokens[0].c_str())-1;
        jj = atoi(tokens[1].c_str())-1;
        vv = atof(tokens[2].c_str());
        if (ii > m){
            Ak = new SparseMat2();
            A.push_back(Ak);
            m++;
        }
        SparseMat2& Akk = *A[ii];
        SparseVec2 aa;
        aa[jj] = vv;
        Akk[jj] = aa;
    }
    m++;
    n = n_raw + m;
    for (int i=0;i<m;i++){
        SparseMat2& Akk = *A[i];
        Akk[n_raw + i][n_raw + i] = 1.0;
    }
    b = new double[m];
    int bi=0;
    while(!b_file.eof()){
        b_file.getline(_line,MAX_LINE);
        if (bi>=m){
            break;
            //cerr<<"b file contains more variables than the number of A's constraints"<<endl;
            //exit(0);
        }
        b[bi] = atof(_line);
        bi++;
    }
    if (bi<m){
        cerr<<"b file contains less variables than the number of A's constraints"<<endl;
        exit(0);
    }

    for (int i=0;i<m;i++){
        SparseVec* aa = new SparseVec();
        C.push_back(aa);
    }
    A_file.close();
    b_file.close();
    c_file.close();
    allocate_prob_a(m);
  /* 
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
    
    // Form b
    b = new double[m];
    for (int i=0;i<m;i++)
        b[i] = 1.0;
    
    prob_a = new double[m];
*/
}

LPProblem::~LPProblem(){
    delete b;
    A.clear();
    C.clear();
}


