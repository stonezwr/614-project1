// my_predictor.h
// This is a perceptron branch predictor created by zhangwenrui@tamu.edu

#include <math.h>
#include <stdint.h>
#include <cstddef>

#define HISTORY_LENGTH	6
#define number_of_perceptrons 2552
#define MASK 		0x000003FF
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128 

//restrict each weight to one byte	
   

class my_update : public branch_update 
{
public:
	unsigned int v[HISTORY_LENGTH]; // record perceptron index
	int yout; // table output
	my_update (void)
	{
		for (int i = 0; i < HISTORY_LENGTH; i++)
			v[i] = 0;
		yout = 0;
	}
};

class my_predictor : public branch_predictor 
{
public:
	int W[H][number_of_perceptrons]; // table of weights	
	uint64_t SG; // hist reg
	uint64_t path; // path reg

	my_predictor (void) 
	{ 
		// initialize the weight table to 0
		for (int i = 0; i < HISTORY_LENGTH; i++) 
		{
			for (int j = 0; j < number_of_perceptrons; j++)
			{
				W[i][j] = 0;
			}
		}
		
		SG = 0;
		
	}
	my_update u;
	branch_info bi;

	// branch prediction
	branch_update *predict (branch_info & b) 
	{
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) 
		{
			u.v[0] = (b.address) % (number_of_perceptrons+5);
			u.yout = W[0][u.v[0]];
			unsigned int seg; 			
			for (int i = 1; i < H; i++) 
			{
				// create segments starting from the most recent history bits and then 
				// moving left
				seg = ((SG ^ (path)) & (MASK << (i-1)*10)) >> (i-1)*10;
				u.v[i] = ((seg) ^ (b.address << 1)) % (number_of_perceptrons-1);
				u.yout += W[i][u.v[i]];	
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
		threshold = (int)(1.89*HISTORY_LENGTH + HISTORY_LENGTH/2);
		int k;
		my_update *mu;
		mu=(my_update *)u;
		if (bi.br_flags & BR_CONDITIONAL)
		{
			if( u->direction_prediction() != taken || abs(mu->yout) < threshold)
			{
				for ( int i = 0; i < HISTORY_LENGTH; i++)
				{
					k=mu->v[i];
					if (taken)
					{
						if (W[i][k] < MAX_WEIGHT)
							W[i][k]++;
					}
					else 
					{
						if (W[i][k] > MIN_WEIGHT)
							W[i][k]--;
					}
				}
			}
		
			// update the hist reg
			SG <<= 1;
			SG |= taken;
			// update the path reg
			// take the last 4 bits of branch addr for path hist reg
			path = bi.address & 0xF;
			path <<= 1;

		}
	}
};
	
