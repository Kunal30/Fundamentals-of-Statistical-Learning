'''
This file is the code for Question 3 of Homework-3
Python Version: 2.7

Generates two moon dataset

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018

Modified by Kunal Suthar
ksuthar1@asu.edu
ASURite ID: 1215112535
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from itertools import *

def gridify(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    print xx.shape
    print yy.shape
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



def genTwoMoons(n=400, radius=1.5, width=0.5, dist=-1):
    rho = radius-width/2 + width*np.random.rand(1,n)
    phi = np.pi*np.random.randn(1,n)
    X = np.zeros((2,n))
    X[0], X[1] = polar2cart(rho, phi)
    id = X[1]<0
    X[0,id] = X[0,id] + radius
    X[1,id] = X[1,id] - dist
    Y = np.zeros(n)
    Y[id] = 1
    return X, Y


def polar2cart(rho, phi):
    x =  rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def main():
    train_data, train_label = genTwoMoons(n=400, radius=1.5, width=0.5, dist=-1)
    validation_data, validation_label = genTwoMoons(n=100, radius=1.5, width=0.5, dist=-1)
    test_data, test_label = genTwoMoons(n=200, radius=1.5, width=0.5, dist=-1)
    colors = ['red','blue']
    train_data=train_data.T
    
    C=[0.001,0.1,10,1000]
    gamma=[0.001,0.01,0.1]
    
    for i,j in product(C,gamma):		    
		print "C=",i,"GAMMA=",j

		SVM_Poly_Model=SVC(C=i,kernel='poly',degree=5,gamma=j)
		SVM_Poly_Model.fit(train_data,train_label)
		valid_accuracy_poly= cross_val_score(SVM_Poly_Model,validation_data.T,validation_label,cv=2)
		print "Polynomial Validation accuracy=",valid_accuracy_poly
		print "Poly accuracy=",SVM_Poly_Model.score(test_data.T,test_label)

		SVM_RBF_Model=SVC(C=i,kernel='rbf',degree=5,gamma=j)
		SVM_RBF_Model.fit(train_data,train_label)
		valid_accuracy_rbf= cross_val_score(SVM_RBF_Model,validation_data.T,validation_label,cv=2)
		print "RBF Validation accuracy=",valid_accuracy_rbf
	    
		Poly_predicts=SVM_Poly_Model.predict(test_data.T)
		RBF_predicts=SVM_RBF_Model.predict(test_data.T)
		
		fig = plt.figure(figsize=(6,4))
		

		X0, X1= test_data[:,0],test_data[:,1]
		xx,yy= gridify(X0,X1)

		#PolyPredicts
		plot_contours(plt,SVM_Poly_Model,xx,yy,cmap=plt.cm.coolwarm,alpha=0.5)
		plt.scatter(test_data[0], test_data[1], c=1-Poly_predicts.T, cmap=clr.ListedColormap(colors))
		plt.show()

		plot_contours(plt,SVM_RBF_Model,xx,yy,cmap=plt.cm.coolwarm,alpha=0.5)
		plt.scatter(test_data[0], test_data[1], c=1-RBF_predicts.T, cmap=clr.ListedColormap(colors))
		plt.show()
		#print "RBF predicts=",SVM_RBF_Model.predict(test_data.T)
		print "RBF accuracy=",SVM_RBF_Model.score(test_data.T,test_label)

    
    
if __name__ == "__main__":
    main()