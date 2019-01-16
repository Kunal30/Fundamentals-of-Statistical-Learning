import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.stats import multivariate_normal
from scipy import random
from scipy.stats import norm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

mean1= np.array([0, 2])
cov1= np.array([[1, 0],[0, 1]])

mean2= np.array([0, 3])
cov2= np.array([[1, 0],[0, 1]])

# feature space for vector x=[x1 x2] over 50 points
x1=np.linspace(-100,100,50)
x2=np.linspace(-100,100,50)
X1,X2= np.meshgrid(x1,x2)
pos = np.dstack((X1, X2)) 


#generating pdf for the first distribution
rv = multivariate_normal(mean1, cov1) 
pdf1=rv.pdf(pos)

#generating pdf for the second distribution
rv2=multivariate_normal(mean2,cov2)
pdf2=rv2.pdf(pos)

#calculating sum of probability of misclassified points in pdf1 and pdf2
error= sum(pdf1)+sum(pdf2)

print 0.5*sum(error)
