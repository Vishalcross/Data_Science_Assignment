'''The objective of this project is to sequntially find how the data interacts with the apriori distribution to create a
   distribution that is a mixture of both the expectaitons as well as the data from the experiment. The data obtained from 
   the experiments modifies our prior distribution to give us a posterior distribution.
   '''
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import gamma
import matplotlib.patches as mpatches
import math

def init():
	'''Hard coded parameters'''
	a=4
	b=6
	sz=150
	mean=0.5
	return a,b,sz,mean

def beta_function(x,a,b):
    '''Manually generating the Beta function with the help of the probability density function'''
    p1 = gamma(a)
    p2 = gamma(b)
    p3 = gamma(a+b)
    p4 = p3/(p1*p2)
    l = []
    for xi in x:
        l.append(p4 * (xi)**(a-1) * (1-xi)**(b-1))
    return l

def main():
    a,b,sz,mean=init()
    results=[]
    data_points=sp.random.binomial(size=sz,n=1,p=mean)
    for data_point in data_points:
        if data_point==1:
            a+=1
        else:
            b+=1
        results+=[(a,b)]
    fig, ax = plt.subplots()
    x = np.arange(0, 1, 0.01)

    ax.set(title = 'Coin Tossing Problem', xlabel = 'Mean', ylabel = 'PDF')
    line, = ax.plot(beta_function(x,1,1), color = 'r', label = 'Sequential Data')

    plt.plot( beta_function(x, results[len(results)-1][0], results[len(results)-1][1]), color='blue', label = 'Complete Data' )

    plt.ylim(0,11)
    #add this - , beta_function(x, results[len(results)-1][0], results[len(results)-1][1]), color='blue'
    def init2():
        line.set_ydata([np.nan] * len(x))
        return line,

    def update(i):
        line.set_ydata(beta_function(x, results[i%len(results)][0], results[i%len(results)][1]) )
        if(i>len(results)):
            ani.event_source.stop()
        return line,

    ani = animation.FuncAnimation(fig, update, init_func = init2, interval = 300, blit = True, repeat_delay = 5000)
    plt.legend()


    ani.save('Animation.mp4')
    plt.show()


if __name__=='__main__':
	main()
