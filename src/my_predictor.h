// my_predictor.h
// This file contains a sample my_predictor class.
// It is a simple 32,768-entry gshare with a history length of 15.
// Note that this predictor doesn't use the whole 8 kilobytes available
// for the CBP-2 contest; it is just an example.
// This is a perceptron branch predictor created by zhangwenrui@tamu.edu
#include <list>
#include <math.h>
#define HISTORY_LENGTH	16
#define number_of_perceptrons 32
#define TABLE_BITS	15
using namespace std;

class my_update : public branch_update {
public:
	unsigned int index;
};

class my_predictor : public branch_predictor {
public:

    double W[number_of_perceptrons][HISTORY_LENGTH];
	my_update u;
	branch_info bi;
	list<int> history;
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
	}
    
	branch_update *predict (branch_info & b) {
		bi = b;
        yout=0;
        list<int>::iterator iter;
        int j;
        if(b.br_flags&&BR_CONDITIONAL){
            perceptron=b.address%number_of_perceptrons;
            yout+=W[perceptron][0];
            for (j=1,iter=history.begin(); iter!=history.end()&&j<HISTORY_LENGTH; j++,iter++) {
                yout+=W[perceptron][j]*(*iter);
            }
            if (yout>=0) {
                u.direction_prediction(true);
                prediction=true;
            }
            else{
                u.direction_prediction(false);
                prediction=false;
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
        list<int>::iterator iter;
        float threshold;
        threshold=2*HISTORY_LENGTH;
        if (bi.br_flags && BR_CONDITIONAL) {
            if (taken!=prediction||fabs(yout)<=threshold) {
                if (taken) {
                    W[perceptron][0]+=1;
                }
                else{
                    W[perceptron][0]-=1;
                }
                for(j=1,iter=history.begin(); iter!=history.end()&&j<HISTORY_LENGTH; j++,iter++) {
                    if ((*iter)!=0) {
                        W[perceptron][j]+=1;
                    }
                    else{
                        W[perceptron][j]-=1;
                    }
                }
            }
        }
        if (history.size()==HISTORY_LENGTH-1) {
            history.pop_back();
        }
        if (taken) {
            history.push_front(1);
        }
        else{
            history.push_front(-1);
        }
    }
};
