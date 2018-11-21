'''The objective of this project is to find the value of pi using method of approximatig the integral of a certain form to a procedure of sampling values
from a PD and finding some statistic of the PD.'''
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from mpl_toolkits.mplot3d import Axes3D

def init():
	'''Hardcoded parameters'''
	n=10 #the number of data points
	r=2 #radius of circle
	exp=7#the required exponent
	powers=[]
	for i in range(1,exp+1,1):
		powers+=[math.pow(10,i)];

	upper_bound=int(math.pow(10,exp))
	return upper_bound,powers,r

def main():
	sum=0
	x_list=[]
	y_list=[]
	z_list=[]
	col_list=[]
	pi_list=[]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('X ')
	ax.set_ylabel('Y ')
	ax.set_zlabel('Z ')

	upper_bound,powers,r=init()
	for i in range(1,upper_bound+1,1):
		x_list.append(np.random.uniform(-r,r))
		y_list.append(np.random.uniform(-r,r))
		z_list.append(np.random.uniform(-r,r))
		if (math.pow(x_list[-1],2)+math.pow(y_list[-1],2)+math.pow(z_list[-1],2))<=math.pow(r,2):
			sum+=1
			col_list.append('blue')
		else:
			col_list.append('red')

		if i in powers:
			pi_list.append(6*sum/i)
			ax.scatter(x_list, y_list, z_list, c=col_list, marker='o')
			plt.savefig("DataPoints:{}".format(i))
			print(pi_list)


if __name__=='__main__':
	main()
