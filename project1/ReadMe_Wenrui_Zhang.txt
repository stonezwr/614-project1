This is the branch prediction program created by Wenrui Zhang, UIN:724005362

In the beginning, as my proposal planed, I tried perceptron neural branch predictor as well as fast path predictor. 

However, I didn't discover much improvement compared to sample predictor. So I turned to a new method Piecewise Linear Branch Prediction.

In the src folder, it has my_predictor_perceptron, my_predictor_fastpath, my_predictor_piecewise for this three method.

my_predictor.h is neural branch predictor with perceptron which has the best performance as I tested.

The following are reference and github for my program:

1.Dynamic branch prediction with perceptrons: http://ieeexplore.ieee.org/document/903263/

2.Fast path-based neural branch prediction: http://ieeexplore.ieee.org/document/1253199/

3.Piecewise linear branch prediction: http://ieeexplore.ieee.org/document/1431572/

Github: https://github.com/stonezwr/614-project1