import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
datasets_dir = '/home/kunal/Desktop/CSE569_HW2/data/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, digit_range=[0, 10], noTrPerClass=100, noTsPerClass=10):
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)
    # plt.imshow(np.reshape(trData[2],(28,28)))
    # plt.show()

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in range(digit_range[0], digit_range[1]):
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl[0], :]
        trY[idx] = trLabels[idl[0]]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl[0], :]
        tsY[idx] = tsLabels[idl[0]]
        count += 1
    

    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = np.squeeze(trX).T
    tsX = np.squeeze(tsX).T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)

    return trX, trY, tsX, tsY


def main():
    trX, trY, tsX, tsY = mnist(noTrSamples=1000,
                               noTsSamples=100, digit_range=[0, 10],
                               noTrPerClass=100, noTsPerClass=10)
    #print trX[234]
    #print trX.shape
    trX=trX.T#np.reshape(trX.T,(1000,784))
    tsX=tsX.T
    #f1=plt.figure(1)
    #plt.imshow(np.reshape(trX[600],(28,28)),cmap=plt.cm.gray)
    #plt.show()
    # print trX[0]
    scalar= StandardScaler()
    trX=scalar.fit_transform(trX)
    pca=PCA(n_components=5)
    
    ld=pca.fit_transform(trX)

    apX=pca.inverse_transform(ld)
    print pca.n_components_
    #plt2=plt.figure(2)
    plt.figure(figsize=(8,4));

	# Original Image
    plt.subplot(1, 2, 1);
    trX=trX*255;
    plt.imshow(trX[1].reshape(28,28),
	              cmap = plt.cm.gray, interpolation='nearest',
	              clim=(0, 255));
    plt.xlabel('784 components', fontsize = 14)
    plt.title('Original Image', fontsize = 20);

	# 154 principal components
    apX=apX*255;
    plt.subplot(1, 2, 2);
    plt.imshow(apX[1].reshape(28, 28),
	              cmap = plt.cm.gray, interpolation='nearest',
	              clim=(0, 255));
    plt.xlabel('154 components', fontsize = 14)
    plt.title('95% of Explained Variance', fontsize = 20);
    plt.show()
if __name__ == "__main__":
    main()
