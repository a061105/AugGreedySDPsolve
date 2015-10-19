#include<stdlib.h>
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include "util.h"

void split(string str, string pattern, vector<string>& tokens){

	tokens.clear();

	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		tokens.push_back(str.substr(i,index-i));

		i = index+1;
	}

	if( tokens.back()=="" )
		tokens.pop_back();
}

double softThd(const double &x,const double  &thd){
	if (x>thd)
		return x-thd;
	else if (x < -thd)
		return x+thd;
	else
		return 0;
}

bool sort_pair (const pair<int,double>& a, const pair<int,double>& b) { return (a.first < b.first); }

double l2_norm_square(double* w, int size){
    double norm=0.0;
    for (int i=0;i<size;i++)
        norm+= w[i] * w[i];
    return norm;
}
double maximum(double* values, int size,int &posi){
	
	double ret = -1e300;
	for(int i=0;i<size;i++){
        if( values[i] > ret ){
			ret = values[i];
            posi = i;
		}
	}
	return ret;
}

int expOverSumExp(double *values, double *prob, int size){
    int posi = -1;
    double max_log = maximum(values,size,posi);
    double sum = 0.0;
    for (int i=0;i<size;i++){
        prob[i] = exp(values[i]-max_log);
    }
    for (int i=0;i<size;i++){
        sum += prob[i];
    }
    for (int i=0;i<size;i++){
        prob[i] /= sum;
    }
    return posi;
}



double maximum(double* values, int size){
	
	double ret = -1e300;
	for(int i=0;i<size;i++){
		if( values[i] > ret ){
			ret = values[i];
		}
	}
	
	return ret;
}

double logSumExp(double* values, int size){
	
	double sum=0.0;
	double max_log = maximum(values, size);
	for(int i=0;i<size;i++){
		sum += exp(values[i] - max_log);
	}
	
	return log(sum) + max_log;
}



double normalize(double* values, int size){
	
	double sum = 0.0;
	for(int i=0;i<size;i++){
		sum += values[i];
	}
	for(int i=0;i<size;i++){
		values[i] /= sum;
	}
	
	return log(sum);
}

void dataToFeatures( vector<vector<pair<int,double> > >& data, int dim,  //intput
		vector<vector<pair<int,double> > >& features //output
		){

	features.clear();
	for(int i=0;i<dim;i++)
		features.push_back(vector<pair<int,double> >());

	for(int i=0;i<data.size();i++){

	    vector<pair<int,double> >* ins = &(data[i]);

		vector<pair<int,double> >::iterator it;
		for(it=ins->begin();it!=ins->end();it++){
			features[it->first].push_back(make_pair(i,it->second));
		}
	}

	//  for (int i=0;i<data.size();i++)
	//      features[dim].push_back(make_pair(i,1.0));
}

void softThd(double* w, vector<int> &act_set, double t_lambda){
	
	double abs_w ;
    int i;
	for(int ii=0; ii<act_set.size();ii++){
		i = act_set[ii];
		abs_w = fabs(w[i]);
		if( abs_w <= t_lambda ){
			
			w[i] = 0.0;

		}else if( w[i] > 0 ){
			
			w[i] = w[i] - t_lambda;
		}else{
			
			w[i] = w[i] + t_lambda;
		}
	}
}


void softThd(double* w, int size, double t_lambda){
	
	double abs_w ;
	for(int i=0; i<size;i++){
		
		abs_w = fabs(w[i]);
		if( abs_w <= t_lambda ){
			
			w[i] = 0.0;

		}else if( w[i] > 0 ){
			
			w[i] = w[i] - t_lambda;
		}else{
			
			w[i] = w[i] + t_lambda;
		}
	}
}

double l1_norm(double* w, int size){
	
	double sum = 0.0;
	for(int j=0;j<size;j++){
		sum += fabs(w[j]);
	}
	return sum;
}

double l1_norm(vector<double>& w){
	
	double sum = 0.0;
	for(int j=0;j<w.size();j++){
		sum += fabs(w[j]);
	}
	return sum;
}

double dot(double* a, double* b,int size)
{
    double sum = 0;
    for (int i=0;i<size;i++)
        sum += a[i]*b[i];
    return sum;
}

double dot(vector<double>& a, vector<double>& b)
{
    if (a.size() != b.size()){
        std::cerr<<"dot product for two vectors with different sizes"<<endl;
        exit(0);
    }
    double sum = 0;
    for (int i=0;i<a.size();i++)
        sum += a[i]*b[i];
    return sum;
}

double l1_norm(double *w, vector<int> &actset){
	
    double sum = 0.0;
	
    for (vector<int>::iterator ii=actset.begin();ii != actset.end();++ii)
		sum += fabs(w[*ii]);
	
    return sum;
}

void shuffle(vector<int>& arr){
	
	int size = arr.size();
	for(int j=0; j<size; j++)
	{
		int i = j+rand()%(size-j);
		swap(arr[i], arr[j]);
	}
}

double sign(double v){
	
	if( v > 0.0 )
		return 1.0;
	else if( v < 0.0 )
		return -1.0;
	else
		return 0.0;
}
