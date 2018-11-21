####################################################################################################################################
Intro:
This assignment contains an implementation for gradient descent.
Dependencies:
It makes use of numpy and pandas and python3
Graphs:
The graphs are generated after running the code for the gradient descent.
Caveat:
We have made use of normalization to reduce the number of iterations required to make the algorithm converge.
Note: 
Keep in mind the location of the files which have been hardcoded to obtain the raw data for the regression.
######################################################################################################################################
Analysis:
1. A large learning rate causes it converge faster, however, too many iterations may cause it to diverge.
2. As the constant for regularization rises, so does the error on the test data for the techniques that have been employed.
3. After a point, the constant reaches a noisy minima. As the values, seem to constantly fluctuate rather than give a smooth behavior.
########################################################################################################################################
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Output Obtained from the Program
***********MLE********************************
[[  1.17942223e-15]
 [ -8.68913756e-01]
 [ -1.69014056e-01]
 [  2.15639519e-02]
 [ -1.37271168e-01]]
The error is:
[[ 0.03761402]]

***********Gradient Descent********************************
[[  1.17644084e-15]
 [ -8.69055257e-01]
 [ -1.68908325e-01]
 [  2.15368774e-02]
 [ -1.37317844e-01]]
The error is:
[[ 0.03761475]]


Value of lambda 1
***********Lasso********************************
[[  8.09978941e-05]
 [ -8.68821143e-01]
 [ -1.68903123e-01]
 [  2.15002936e-02]
 [ -1.37059752e-01]]
The error is:
0.0376138451975

***********Ridge********************************
[[  1.17311061e-15]
 [ -8.67715247e-01]
 [ -1.69736201e-01]
 [  2.18244088e-02]
 [ -1.36838814e-01]]
The error is:
0.0376089618321
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@