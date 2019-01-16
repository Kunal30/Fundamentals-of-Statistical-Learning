import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.stats import multivariate_normal
from scipy import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import spline
d=2 # d dimension

mean1= np.array([0, 2])
mean1=mean1.transpose()
#mean1=mean1.ravel()
#print(mean1)

cov1= np.array([[1, 0],[0, 1]])

xlist,pdf1,pdf2=[],[],[] # for storing the results of the 1000 points
x_1,x_2=[],[]
for i in range(0,1000):
	x=np.random.random_integers(low=0,high=10,size=d) # d dimensional x feature vector
	#print(x)
	y=multivariate_normal.pdf(x,mean=mean1,cov=cov1) # class conditional probability with mean mu and covariance matrix 
	xlist.append(x)
	pdf1.append(y)

mean2= np.array([0, 3])
mean2=mean2.transpose()

cov2=np.array([[1,0],[0,1]])
for x in xlist:	
	x_1.append(x[0])
	x_2.append(x[1])
	y=multivariate_normal.pdf(x,mean=mean2,cov=cov2) # class conditional probability with mean mu and covariance matrix 	
	pdf2.append(y)

print "PDF1=",pdf1
print "PDF2=",pdf2

# plt.plot(x_1,pdf1)
# plt.show()
# plt.plot(xlist,pdf2,'x')
# plt.show()