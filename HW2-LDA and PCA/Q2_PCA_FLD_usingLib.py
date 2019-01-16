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
datasets_dir = '/home/kunal/Desktop/CSE569_HW2/data/'


def calculate_percentVarianceRetained(pca1,pca2):
	return (sum(pca1.explained_variance_)/sum(pca2.explained_variance_))*100;

def main():

	#generating mentioned training & testing samples
    trX, trY, tsX, tsY = mnist(noTrSamples=1000,
                               noTsSamples=100, digit_range=[0, 10],
                               noTrPerClass=100, noTsPerClass=10) 
    
    #taking transpose to fit the required matrix format
    trX=trX.T
    tsX=tsX.T
 
    #pca object to reduce image to 100 components
    pca=PCA(n_components=100)

    #pca object having 784 components
    pca2=PCA(n_components=784)
    
    #scaling the input to fit the transform
    scalar= StandardScaler()
    trX=scalar.fit_transform(trX)

    #fitting data into pca of 100 components
    eTx=pca.fit_transform(trX)  
    
    #Transforming reduced data back to its original space
    #for plotting and comparison with original 
    apX=pca.inverse_transform(eTx)
    
    #fitting data into pca of 784 components
    ld2=pca2.fit_transform(trX)    
    
    #Transforming reduced data back to its original space
    #for plotting and comparison with original 
    apX2=pca2.inverse_transform(ld2)
    
    #calculating retained % variance    
    percVretained=calculate_percentVarianceRetained(pca,pca2);
    percVretained=round(percVretained,2)
    print percVretained

    #printing the covariance matrix
    cov=getCovariance(eTx)
    

    #Performing PCA Whitening
    Xpca_w=perform_PCA_Whitening(pca,eTx,cov)

    #Transforming reduced data back to its original space
    #for plotting and comparison with original 
    apX3=pca.inverse_transform(Xpca_w.T)        
    cov2=getCovariance(Xpca_w)    
    #displayImgs(apX,apX3,percVretained)

    #Performing ZCA Whitening
    Xzca_w=perform_ZCA_Whitening(pca,Xpca_w,cov)
    cov3=getCovariance(Xzca_w)
    displayImgs(apX3,Xzca_w.T,percVretained)

def perform_ZCA_Whitening(pca,Xpca_w,cov):
	 eigen_vectors=pca.components_
	 return np.dot(eigen_vectors.T,Xpca_w)

def perform_PCA_Whitening(pca,eTx,cov):
	eigen_vectors=pca.components_
	eigen_values=pca.explained_variance_	
	x=np.dot(eigen_vectors.T,eTx.T)
	for i in range(0,100):
		if(eigen_values[i]==0):
			eigen_vectors[i]=eigen_vectors[i]/math.sqrt(eigen_values[i]+1e-3);
		else:
			eigen_vectors[i]=eigen_vectors[i]/math.sqrt(eigen_values[i]);

	return np.dot(eigen_vectors,x)


def getCovariance(apX):
	covarianceMat=np.cov(apX)	
	plt.matshow(covarianceMat*255,cmap=plt.cm.hsv) # hsv, RdBu
	plt.show()

def displayImgs(trX,apX,percVretained):
	
	i=0
	while i < 999:
		# Original Image
	    plt.subplot(1, 2, 1);
	    trX=trX*255;
	    plt.imshow(trX[i].reshape(28,28),cmap = plt.cm.gray, 
	    	interpolation='nearest',
		              clim=(0, 255));
	    plt.xlabel('100 components', fontsize = 14)
	    plt.title('X PCA Whitening', fontsize = 20);

		# 100 principal components
	    apX=apX*255;
	    plt.subplot(1, 2, 2);
	    plt.imshow(apX[i].reshape(28, 28),cmap = plt.cm.gray,
	    	interpolation='nearest',
		              clim=(0, 255));
	    plt.xlabel('100 components', fontsize = 14)
	    plt.title('X ZCA Whitening', fontsize = 20);
	    plt.show()
	    i+=100

if __name__ == "__main__":
    main()
