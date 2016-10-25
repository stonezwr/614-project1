// my_predictor.h
// This is a perceptron branch predictor created by zhangwenrui@tamu.edu

#include <math.h>
#include <stdint.h>
#include <cstddef>

#define HISTORY_LENGTH	16
#define ADDRESS 32
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128 //restrict each weight to one byte	


class my_update : public branch_update 
{
public:
	int yout; // table output
	my_update (void)
	{
		yout = 0;
	}
};

class my_predictor : public branch_predictor 
{
public:
	int W[ADDRESS][ADDRESS][HISTORY_LENGTH+1]; // table of weights	
	bool GHR[HISTORY_LENGTH];
	unsigned int GA[HISTORY_LENGTH];
    int perceptron;

	my_predictor (void) 
	{ 
		// initialize the weight table to 0
		for (int i = 0; i < ADDRESS; i++) 
		{
			for (int j = 0; j < ADDRESS; j++)
			{
				for(int k=0; k<=HISTORY_LENGTH;k++){
					W[i][j][k] = 0;
				}
			}
		}
		for(int i=0;i<HISTORY_LENGTH;i++){
			GHR[i]=true;
			GA[i]=0;
		}
		
	}
	my_update u;
	branch_info bi;

	// branch prediction
	branch_update *predict (branch_info & b) 
	{
		bi = b;
		unsigned int add;
		if (b.br_flags & BR_CONDITIONAL) 
		{
			perceptron=(b.address) % ADDRESS;
			u.yout = W[perceptron][0][0];
			for (int i = 1; i<=HISTORY_LENGTH; i++) 
			{
				if(GA[i-1]!=0){
					add=GA[i-1]%ADDRESS;
					if(GHR[i-1]){
						u.yout+=W[perceptron][add][i];
					}
					else{
						u.yout-=W[perceptron][add][i];
					}
				}
            }
			if (u.yout >= 0) 
			{
				u.direction_prediction (true);
			}
			else
			{
				u.direction_prediction (false);
			}
		} 
		else
		{
			u.direction_prediction (true); // unconditional branch
		}
		u.target_prediction (0); // not computing target address
		return &u;
	}

	// training algorithm
	void update (branch_update *u, bool taken, unsigned int target) 
	{
		float threshold;
		unsigned int add;
		threshold = (int)(1.94*HISTORY_LENGTH + 14); //threshold get from paper
		my_update *mu;
		mu=(my_update *)u;
		if (bi.br_flags & BR_CONDITIONAL)
		{
			if( u->direction_prediction() != taken || abs(mu->yout) < threshold)
			{
				if (taken)
				{
					if (W[perceptron][0][0] < MAX_WEIGHT)
						W[perceptron][0][0]++;
				}
				else 
				{
					if (W[perceptron][0][0] < MIN_WEIGHT)
						W[perceptron][0][0]--;
				}
				for ( int i = 1; i <= HISTORY_LENGTH; i++)
				{
					if(GA[i-1]!=0){
						add=GA[i-1]%ADDRESS;
						if(taken==GHR[i-1])
						{
							if (W[perceptron][add][i] < MAX_WEIGHT)
								W[perceptron][add][i]++;
						}
						else 
						{
							if (W[perceptron][add][i] > MIN_WEIGHT)
								W[perceptron][add][i]--;
						}
					}
				}
			}	
			for(int i=HISTORY_LENGTH-1;i>0;i--){
				GA[i]=GA[i-1];
				GHR[i]=GHR[i-1];
			}
			GA[0]=bi.address;
			GHR[0]=taken;
		}
	}
};
	
