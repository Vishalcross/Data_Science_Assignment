'''The objective of this project is to find the value of pi using method of approximatig the integral of a certain form to a procedure of sampling values 
from a PD and finding some statistic of the PD.'''
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt 
import math 




def main():
    n = 10   #intialization for the no of sample points
    r = 2    # range of the x and y 
    pi = []  
    i = 0
    while n<=math.pow(10,7):
        xs = []
        ys = []
        col = []
        sum = 0
        for i in range(n): 
            xs.append(np.random.uniform(-r,r))
            ys.append(np.random.uniform(-r,r))
            if math.pow(xs[i],2)+math.pow(ys[i],2)<=math.pow(r,2):
                sum += 1
                col.append('blue')
            else:
                col.append('red')
       
        pi.append((4*sum)/n)
        plt.scatter(xs, ys, c=col, s=5, linewidth=0)
        plt.show()
        '''  
        col = np.where(math.pow(xs,2)+math.pow(ys,2)<=math.pow(r,2),'b')
        
        
        '''
        
        print(pi)

        n = n*10 
        
        sum = 0 
    
        i = 0 

if __name__=='__main__':
	main()
    

