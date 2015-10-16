#ifndef SEQ_LABEL
#define SEQ_LABEL

#include <set>
#include "problem.h"
#include <map>

typedef vector<pair<int,double> > Feature;

class Seq{
	public:
	Seq(){ T = 0; }
	int T;
	int* labels; //1*T
	vector<Feature*> features; // 1*T
};

class SeqFactor{
	public:
	double** factor_xy; //1*T array of "1*|Y| table"
};

class SeqMarg{
	public:
	double** forward_msg; //(exclusive)
	double** forward_inc_msg; //(incusive)
	double** backward_msg; //(inclusive)
	double** marginal;
};

class SeqMax{
    public:
        double** alpha; //T by K
        int** traceBack; //(T-1) by K
        int* maxLabel; //T
};

class SeqLabelProblem:public Problem{
	
	/* w: d_expand *|Y| + |Y|*|Y| ( row:(t), col:(t+1), row-major store )
	 *
	 * fvalue: N*T*|Y| + |Y|*|Y| (no bi-fea)
	 * 	   N*T*|Y| +  NT*|Y|*|Y| (with bi-fea)
	 */

	public:
	//int raw_d;
	
	//Functions for "Problem"
	void compute_fv_change(double* w_change, vector<int>& act_set, //input
			vector<pair<int,double> >& fv_change); //output
	
	void update_fvalue(vector<pair<int,double> >& fv_change, double scalar);
	
	void grad( vector<int>& act_set, //input
			vector<double>& g);//output
    
	double fun();
	
    void test_accuracy(const char* output_file);
    
    double train_accuracy();
    
    void readProblem(char* data_file);
    void readProblem(char* model_file, char* data_file);
    
    //Function specific to Sequence Labeling Problem
	SeqLabelProblem(char* data_file);
	SeqLabelProblem(char* model_file, char* data_file);
	
    int numLabel;
	int bi_offset_w;
	int bi_offset_fv;
	int T_total;	
	map<string,int> label_index_map;
	map<int,string> label_name_map;

	//store sequences 
	vector<Seq> data; //N
	vector<int> labels; //T_total
	vector<Feature> data_inv; //for each fea j, store set factors(j): the affected factors (n,t) by fea j
	void build_data_inv();
    set<int> sample_updated;
    bool bi_factor_updated;
	//factor values
	SeqFactor* factors; //N
	double* factor_yy;
	
    void compute_fvalue();
	//calibrated forward (exlusive), backward (inclusive) messages
	SeqMarg* seqMargs; // N
    SeqMax* seqMaxs;
    void inferMarg();
    void inferMax();

	double** forward_msg; //T_total * |Y|
	double** backward_msg; 
	double** forward_inc_msg; 
	double** marginal;
	double logZ;
	
	//feature template
	vector<int> uni_fea_template;
	int d_expand;
	int compute_d_expand(int raw_d);
	void feature_expand(Seq& seq, int t, Feature& fea_exp);
};

#endif
