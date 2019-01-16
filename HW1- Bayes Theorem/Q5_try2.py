import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.stats import multivariate_normal
from scipy import random
from scipy.stats import norm

x=np.linspace(0,10,1000)
pdf1=norm(0,2).pdf()

plt.plot(x,pdf1)
plt.show()