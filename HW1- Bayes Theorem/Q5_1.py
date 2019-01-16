# import matplotlib.pyplot as plt
# import math
# import numpy as np
# import os
# from scipy.stats import multivariate_normal
# from scipy import random

# d=1 # d dimension

# # creating random covariance matrix
# matrixSize = d 
# A = random.rand(matrixSize,matrixSize)
# #A= A*10
# covariance = 1#np.dot(A,A.transpose())

# mu= 0 #random.rand(d) #mean 

# xlist=np.linspace(-3,3,1000)
# #ylist=[]
# #xlist,ylist=[],[] # for storing the results of the 1000 points
# # for x in xlist:
# # 	#x=np.random.rand(d) # d dimensional x feature vector
# # 	y=multivariate_normal.pdf(x,mean=mu,cov=covariance) # class conditional probability with mean mu and covariance matrix 
# # 	#xlist.append(x)
# # 	ylist.append(y)	

# pdf1=multivariate_normal.pdf(xlist,mean=mu,cov=covariance)
# #print(xlist)
# print(pdf1)

# #print "mu=",mu
# plt.plot(xlist,pdf1,'-k',color='r')
# plt.axis('equal')
# plt.show()
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.stats import multivariate_normal
from scipy import random

d=1 # d dimension

# creating random covariance matrix
matrixSize = d 
A = random.rand(matrixSize,matrixSize)
covariance = np.dot(A,A.transpose())

mu= random.rand(d) #mean 


xlist,ylist=[],[]

# generating 1000 sample points in the range of (0,1)
for i in range(0,1000):
	x=np.random.rand(d) # d dimensional x feature vector
	y=multivariate_normal.pdf(x,mean=mu,cov=covariance) # class conditional probability with mean mu and covariance matrix 
	xlist.append(x)
	ylist.append(y)	

# 
plt.plot(xlist,ylist,'x')
plt.show()