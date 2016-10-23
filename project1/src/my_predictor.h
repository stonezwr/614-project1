// my_predictor.h
// This is a perceptron branch predictor created by zhangwenrui@tamu.edu
#include <list>
#include <vector>
#include <math.h>
#include <stdint.h>
#include <cstddef>
#define HISTORY_LENGTH	6
#define number_of_perceptrons 2552
#define MASK        0x00000001
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

using namespace std;

class my_update : public branch_update {
public:
	unsigned int v[HISTORY_LENGTH]; //represent the last h perceptron;
    int yout;
    my_update(){
        for (int i = 0; i < HISTORY_LENGTH; ++i)
        {
            v[i]=0;
        }
        yout=0;
    }
};

class my_predictor : public branch_predictor {
public:

    int W[number_of_perceptrons][HISTORY_LENGTH+1];
    vector<int> SR;
    vector<int> R;
	my_update u;
	branch_info bi;
	uint64_t G;
    uint64_t SG;
    int perceptron;

	my_predictor(){
        int i,j;
        for (i=0; i<number_of_perceptrons; i++) {
            for (j=0; j<HISTORY_LENGTH; j++) {
                W[i][j]=0;
            }
        }
        SR.resize(HISTORY_LENGTH+1);
        R.resize(HISTORY_LENGTH+1);
        G=0;
        SG=0;
	}
    
	branch_update *predict (branch_info & b) {
		bi = b;
        int j;
        int k;
        if(b.br_flags&&BR_CONDITIONAL){
            perceptron=b.address%number_of_perceptrons;
            u.v[0]=perceptron;
            u.yout=W[perceptron][0]+SR[HISTORY_LENGTH];
            if (u.yout>=0) {
                u.direction_prediction(true);
            }
            else{
                u.direction_prediction(false);
            }
            for (j=1; j<=HISTORY_LENGTH; j++) {
                k=HISTORY_LENGTH-j;
                if (u.direction_prediction())
                {
                    SR[k+1]=SR[k]+W[perceptron][j];
                }
                else{
                    SR[k+1]=SR[k]-W[perceptron][j];
                }
            }
            SR[0]=0;
            SG <<= 1;
            SG |= u.direction_prediction();
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
        int threshold;
        my_update *mu;
        mu=(my_update *)u;
        threshold=(int)1.93*HISTORY_LENGTH+14;
        if (bi.br_flags && BR_CONDITIONAL) {
            if (taken!=u->direction_prediction()||abs(mu->yout)<=threshold) {
                if (taken) {
                    W[perceptron][0]+=1;
                }
                else{
                    W[perceptron][0]-=1;
                }
                for(j=1; j<=HISTORY_LENGTH; j++) {
                    k=(*mu).v[j-1];
                    if (SG&(MASK<<(j-1))) {
                        if (W[k][j]<MAX_WEIGHT)
                        {
                            W[k][j]+=1;
                        } 
                    }
                    else{
                        if(W[k][j]>MIN_WEIGHT){
                            W[k][j]-=1;
                        }
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
        G <<= 1;
        G |= taken;
        if (taken!=u->direction_prediction())
        {
            SG=G;
            SR=R;
        }
    }
};
