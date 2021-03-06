#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <vector>
#include <map>
#include <stdlib.h>
#include <cmath>
using namespace std;

const int MAX_LINE = 100000000;
typedef vector<pair<int,double> > SparseVec;
typedef map<int,double> SparseVec2;
typedef vector<SparseVec*> SparseMat;
typedef map<int,SparseVec> SparseMat2;

class ScoreComp{
	
	public:
	ScoreComp(double* _score):score(_score){
	}
	bool operator()(int i, int j){
		return score[i] > score[j];
	}
	private:
	double* score;
};

double normalize(SparseVec* ins){
	
	double norm2 = 0.0;
	for(SparseVec::iterator it = ins->begin();it!=ins->end();it++){
		norm2 += it->second*it->second;
	}
	norm2 = sqrt(norm2);
	
	for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
		it->second /= norm2;

	return norm2;
}

double dot(SparseVec* ins1, SparseVec* ins2){
	
	double sum = 0.0;
	vector<pair<int,double> >::iterator it  = ins1->begin();
	vector<pair<int,double> >::iterator it2 = ins2->begin();
	while( it!=ins1->end() && it2!=ins2->end() ){
		
		if( it->first < it2->first ){
			it++;
		}else if( it2->first < it->first ){
			it2++;
		}else{
			sum += it->second * it2->second;
			it++;
			it2++;
		}
	}

	return sum;
}

void random_uniform(int range, int num, vector<int>& random_numbers){
	
	random_numbers.clear();
	for(int i=0;i<num;i++){
		random_numbers.push_back( rand()%range );
	}
}

int argmin( double* values, int size ){
	
	double min_val = 1e300;
	int min_ind = -1;
	for(int i=0;i<size;i++){
		if( values[i] < min_val ){
			min_val = values[i];
			min_ind = i;
		}
	}
	return min_ind;
}

// v1 += c*v2
void vadd( double* v1, double c, double* v2, double* v3, int size){
	
	for(int i=0;i<size;i++)
		v3[i] = v1[i] + c*v2[i];
}

void vadd( vector<double>& v1, double c, vector<double>& v2, vector<double>& v3, int size){
	
	for(int i=0;i<size;i++)
		v3[i] = v1[i] + c*v2[i];
}

SparseMat* ones(int R, int C){
	
	SparseMat* mat = new SparseMat();
	for(int i=0;i<R;i++){
		mat->push_back(new SparseVec());
		for(int j=0;j<C;j++)
			mat->at(i)->push_back(make_pair(j,1.0));
	}
	return mat;
}

/** Sample from Multi-nouli distribution defined by the potential
 *  potential must be positive.
 */
int sample_multinouli(int size, double* potential){
	
	double* cumul_pot = new double[size];
	cumul_pot[0] = potential[0]; 
	for(int i=1;i<size;i++)
		cumul_pot[i] = cumul_pot[i-1] + potential[i];
	
	double r = ((double)rand()/RAND_MAX)*cumul_pot[size-1];
	int i;
	for(i=0; i<size && cumul_pot[i]<r ;i++);
	
	delete[] cumul_pot;
	
	return i;
}

void sparseToDense( SparseVec* sv, double* v, int size ){
	
	for(int i=0;i<size;i++)
		v[i] = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++)
		v[it->first] = it->second;
}

double sv_max( SparseVec* sv ){
	
	double max_val = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++){
		if( it->second > max_val )
			max_val = it->second;
	}
	return max_val;
}

int num_nonzeros(SparseMat2& WT){
	int nnz = 0;
	for(SparseMat2::iterator it=WT.begin(); it!=WT.end(); it++){
		nnz += it->second.size();
	}
	return nnz;
}

int num_nonzero_rows(SparseMat2& WT){
	int count = 0;
	for(SparseMat2::iterator it=WT.begin(); it!=WT.end(); it++){
		if( it->second.size() != 0 )
			count++;
	}
	return count;
}

void printSparseMat(SparseMat2& WT, ostream& out){
	
	for(SparseMat2::iterator it=WT.begin(); it!=WT.end(); it++){
		SparseVec* sv = &(it->second);
		if( sv->size() == 0 )
			continue;

		out << it->first << " ";
		for(SparseVec::iterator it2=sv->begin(); it2!=sv->end(); it2++)
			out << it2->first << ":" << it2->second << " ";
		out << endl;
	}
}

double rand01(){
	
	return ((double)rand()/RAND_MAX);
}

void readVec(ifstream& fin, double* vec, int size){
	
	double val;
	for(int i=0;i<size;i++){
		fin >> val;
		vec[i] = val;
	}
}
#endif
