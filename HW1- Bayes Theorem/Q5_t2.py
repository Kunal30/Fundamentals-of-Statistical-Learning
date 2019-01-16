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

x1, x2= np.mgrid[-3:4:.1, -3:7:.1] # feature space for vector x=[x1 x2]
pos = np.dstack((x1, x2)) 

#generating pdf for the first distribution
rv = multivariate_normal(mean1, cov1) 
pdf1=rv.pdf(pos)

#generating pdf for the second distribution
rv2=multivariate_normal(mean2,cov2)
pdf2=rv2.pdf(pos)

#for displaying the pdfs

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x1, x2, pdf1, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=cm.viridis)
cset = ax.contourf(x1, x2, pdf1, zdir='z', offset=-0.15, cmap=cm.viridis)
ax.plot_surface(x1, x2, pdf2, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=cm.coolwarm)
cset = ax.contourf(x1, x2, pdf2, zdir='z', offset=-0.15, cmap=cm.coolwarm)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()