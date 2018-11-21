'''The objective of this project is to find the value of pi using method of approximatig the integral of a certain form to a procedure of sampling values 
from a PD and finding some statistic of the PD.'''
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt 
import math 

def init():
	'''Hardcoded parameters'''
	n=10 #the number of data points
	r=2 #radius of circle
	exp=7 #the required exponent
	powers=[]
	for i in range(1,7,1):
		powers+=[math.pow(10,i)];

	upper_bound=int(math.pow(10,7))
	return upper_bound,powers,r

def main():
	sum=0
	x_list=[]
	y_list=[]
	col_list=[]
	pi_list=[]
	upper_bound,powers,r=init()
	for i in range(1,upper_bound,1):
		x_list.append(np.random.uniform(-r,r))
		y_list.append(np.random.uniform(-r,r))

		if (math.pow(x_list[-1],2)+math.pow(y_list[-1],2))<=math.pow(r,2):
			sum+=1
			col_list.append('blue')
		else:
			col_list.append('red')

		if i in powers:
			pi_list.append(4*sum/i)
			plt.scatter(x_list,y_list,c=col_list,s=5,linewidth=0)
			plt.show()
			print(pi_list)

if __name__=='__main__':
	main()