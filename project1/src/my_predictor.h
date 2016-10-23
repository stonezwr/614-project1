// my_predictor.h
// This file contains a sample my_predictor class.
// It is a simple 32,768-entry gshare with a history length of 15.
// Note that this predictor doesn't use the whole 8 kilobytes available
// for the CBP-2 contest; it is just an example.
// This is a perceptron branch predictor created by zhangwenrui@tamu.edu
#include <list>
#include <vector>
#include <math.h>
#define HISTORY_LENGTH	256
#define number_of_perceptrons 1024
using namespace std;

class my_update : public branch_update {
public:
	unsigned int index;
};

class my_predictor : public branch_predictor {
public:

    double W[number_of_perceptrons][HISTORY_LENGTH+1];
    vector<int> SR;
    vector<int> R;
	my_update u;
	branch_info bi;
	list<int> G;
    list<int> SG;
    vector<int> v;      //represent the last h perceptron;
    double yout;
    int perceptron;
    bool prediction;


	my_predictor(){
        int i,j;
        for (i=0; i<number_of_perceptrons; i++) {
            for (j=0; j<HISTORY_LENGTH; j++) {
                W[i][j]=0;
            }
        }
        SR.resize(HISTORY_LENGTH+1);
        R.resize(HISTORY_LENGTH+1);
        for(i=0;i<HISTORY_LENGTH+1;i++){
            SR[i]=0;
            R[i]=0;
        }
	}
    
	branch_update *predict (branch_info & b) {
		bi = b;
        yout=0;
        list<int>::iterator iter;
        int i,j;
        int k;
        v.resize(HISTORY_LENGTH);
        if(b.br_flags&&BR_CONDITIONAL){
            perceptron=b.address%number_of_perceptrons;
            if (v.size()>=HISTORY_LENGTH)
            {
                for (i=HISTORY_LENGTH-1;i>0; i--)
                {
                    v[i]=v[i-1];
                }
            }
            else{
                for (i=v.size();i>0; i--)
                {
                    v[i]=v[i-1];
                }
            }
            v[0]=perceptron;
            yout=W[perceptron][0]+SR[HISTORY_LENGTH];
            if (yout>=0) {
                u.direction_prediction(true);
                prediction=true;
            }
            else{
                u.direction_prediction(false);
                prediction=false;
            }
            for (j=1; j<=HISTORY_LENGTH; j++) {
                k=HISTORY_LENGTH-j;
                if (prediction)
                {
                    SR[k+1]=SR[k]+W[perceptron][j];
                }
                else{
                    SR[k+1]=SR[k]-W[perceptron][j];
                }
            }
            SR[0]=0;
            if(SG.size()>=HISTORY_LENGTH){
                SG.pop_back();
            }
            if(prediction){
                SG.push_front(1);
            }
            else{
                SG.push_front(0);
            }
        }
        else{
            u.direction_prediction(false);
        }
        u.target_prediction(0);
        return &u;
        
	}

	void update (branch_update *u, bool taken, unsigned int target) {
        int j;
        int k;
        list<int>::iterator iter;
        float threshold;
        threshold=1.93*HISTORY_LENGTH+14;
        if (bi.br_flags && BR_CONDITIONAL) {
            if (taken!=prediction||fabs(yout)<=threshold) {
                if (taken) {
                    W[perceptron][0]+=1;
                }
                else{
                    W[perceptron][0]-=1;
                }
                for(j=1,iter=SG.begin(); iter!=SG.end()&&j<=HISTORY_LENGTH; j++,iter++) {
                    k=v[j-1];
                    if ((*iter)!=0) {
                        W[k][j]+=1;
                    }
                    else{
                        W[k][j]-=1;
                    }
                }
            }
            perceptron=bi.address%number_of_perceptrons;
            for (j=1; j<=HISTORY_LENGTH; j++) {
                k=HISTORY_LENGTH-j;
                if (taken)
                {
                    R[k+1]=R[k]+W[perceptron][j];
                }
                else{
                    R[k+1]=R[k]-W[perceptron][j];
                }
            }
            R[0]=0;
        }
        if (G.size()>=HISTORY_LENGTH) {
            G.pop_back();
        }
        if (taken) {
            G.push_front(1);
        }
        else{
            G.push_front(0);
        }
        if (taken!=prediction)
        {
            SG=G;
            SR=R;
        }
    }
};
