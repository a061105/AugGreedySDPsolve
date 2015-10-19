#ifndef UTIL
#define UTIL

#include<vector>
#include<map>
#include<string>
#include<cmath>
#include <iomanip>      // std::setprecision
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

void split(string str, string pattern, vector<string>& tokens);

double maximum(double* values, int size);

double maximum(double* values, int size,int &posi);

int expOverSumExp(double *values, double *prob, int size);

double logSumExp(double* values, int size);

double normalize(double* values, int size);

void dataToFeatures( vector<vector<pair<int,double> > >& data, int dim,  //intput
		vector<vector<pair<int,double> > >& features //output
        );	

void softThd(double* w, vector<int> &act_set, double t_lambda);
void softThd(double* w, int size, double t_lambda);
double softThd(const double &x,const double  &thd);

double l1_norm(double* w, int size);

double l1_norm(vector<double>& w);

double l1_norm(double *w, vector<int> &actset);
double dot(double* a, double* b, int size);
double dot(vector<double>& a, vector<double>& b);
double l2_norm_square(double* w,int size);
bool sort_pair (const pair<int,double>& a, const pair<int,double>& b);
void shuffle(vector<int>& arr);

double sign(double v);

#endif
