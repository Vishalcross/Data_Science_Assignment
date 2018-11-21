####################################################################################################################################
Intro:
This assignment contains an implementation for K nearest neighbours
Dependencies:
It makes use of numpy and pandas and python3
Graphs:
The graphs are generated after running the code for the K nearest neighbours in the following forms
	1. Rolling window
	2. Recursive window
	Each in turn have been implemented for normalized and not normalized
Caveat:
We have made use of weighed averages to reduce the value of error obtained. For non-normalized, one by the cartesian distance is used
as the weight and for normalization the value of the cosine is used for the weight.
Note: 
Keep in mind the location of the files which have been hardcoded to obtain the raw data for the prediction.
In addition, this code has a large running time, so run the modules individually or as different processes.
The graphs have been saved rather than being rendered on the screen.
Graphs:
They have been stored in a folder named graphs. However, they are created in the working directory by default.
Large graphs go from k=1 to k=2^17 all the graphs only go till 2^9.
The graphs are labelled according to the following convention-
	normalised_recursive and normalised_rolling contain the images for the normalised versions of recursive and rolling window.
	recursive_window and rolling_window contain the images for recursive and rolling windows respectively.
	!!!!WARNING!!! 
	The large windows are not generated as they have a unsustainably large running time and the global minima is not visible in the
	large graph. For creating such graphs change the sentinel value[currently 10] in the while loop.
######################################################################################################################################
Analysis:
1. PM 25 Error is always greater than the PM10 error
2. Recursive and rolling windows as well as their normalized counterparts give similar values for error and the predictions
3. Optimal value for k seems to be close to 2^6 and 2^9 neighbours, approximately one day to 3 weeks gives a reasonable value for 
   the prediction.
4. We see that the value for the error falls for k=1 as compared to k=4 for PM10 in non-normalised recursive and rolling windows. 
########################################################################################################################################
