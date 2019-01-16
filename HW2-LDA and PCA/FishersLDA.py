import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import cv2
import math
from load_mnist_updated_4Oct2018 import mnist
from load_mnist_updated_4Oct2018 import one_hot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib as mpl
from matplotlib import colors

datasets_dir = '/home/kunal/Desktop/CSE569_HW2/data/'

def main():

	#generating mentioned training & testing samples
    trX, trY, tsX, tsY = mnist(noTrSamples=1000,
                               noTsSamples=100, digit_range=[0, 10],
                               noTrPerClass=100, noTsPerClass=10) 
    
    #taking transpose to fit the required matrix format
    trX=trX.T
    tsX=tsX.T
    tsY=tsY.T

    #generating training images and labels of only 0's and 1's
    trX_01=trX[0:200]
    trY_01=trY[0][0:200]
    trX_01=np.array(trX_01)
    trY_01=np.array(trY_01)    
    
    # Empty list for finding out test images and labels of only 0s
    # and 1s
    tsX_01=[]    
    tsY_01=[]    
    
    #pca object to reduce image to 100 components
    pca=PCA(n_components=100)
    
    #scaling the input to fit the transform
    scalar= StandardScaler()
    trX_01=scalar.fit_transform(trX_01)    
    tsX=scalar.fit_transform(tsX)

    #fitting data into pca of 100 components    
    eTx=pca.fit_transform(trX_01)      
    
    # Fitting training images to Fisher's 
    # Linear Discriminant space
    fLD= LinearDiscriminantAnalysis()        
    fLD.fit(eTx,trY_01)
    
    #reducing test set to 100 components    
    reduced_tsX=pca.fit_transform(tsX)        
    
    # collecting only the 0's and 1's from the 
    # test images
    for i in range(0,100):
     	if tsY[i]==0 or tsY[i]==1:
     		tsX_01.append(reduced_tsX[i])
     		tsY_01.append(tsY[i])
    
    #Converting the list object to an nparray
    tsX_01=np.array(tsX_01)
    tsY_01=np.array(tsY_01) 		 		
    
    #Predicting results on the test-set
    results=fLD.predict(tsX_01)
    
    #Results
    print results    
    print fLD.score(tsX_01,tsY_01)*100  
    
if __name__ == "__main__":
    main()    