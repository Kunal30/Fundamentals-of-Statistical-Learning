import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.stats import multivariate_normal
from scipy import random


d=2 # d dimension

# creating random covariance matrix
#matrixSize = d 
#A = random.rand(matrixSize,matrixSize)
#A= A*10
covariance1 = np.matrix('1 0;0 1') #np.dot(A,A.transpose())


mu1= np.matrix('0 2') #random.random_integers(100,size=(d)) #mean 
mu1=mu1.transpose()
print mu1
#x,y=np.random.multivariate_normal(mu,covariance,5000).T
#count, bins, ignored = plt.hist(s, 30, density=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
x=np.random.random_integers(10,size=(d)) # d dimensional x feature vector
x=np.matrix(x)
x=x.transpose()
print x
pdf1=multivariate_normal.pdf(x,mean=mu1,cov=covariance1); # class conditional probability with mean mu and covariance matrix 

print(pdf1)

#fig1=plt.figure()
#ax=fig1.add_subplot(111)
#plt.plot(x,y,'o',color='r')
#plt.show()	
#plt.plot(x,y,'o',color='r')
#plt.axis('equal')
#plt.show()