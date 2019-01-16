'''
This file is the code for Question 1 of Homework-3
Python Version: 2.7

This file implements a two layer neural network for a binary classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018

Modified by: Kunal Suthar
ksuthar1@asu.edu
ASURite ID:1215112535
'''
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    # Regularization 
    #maxval= np.amax(Z)
    # Z = Z-maxval

    A = 1/(1+np.exp(-Z))
    #print "A=",A
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    #print dA.shape
    # print cache
    #print cache["Z"]#.shape
    
    a,tempcache=sigmoid(cache["Z"])    
    der= np.multiply(a,1-a)
    dZ = np.multiply(dA,der)
    
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE

    parameters = {}
    W1=np.random.randn(n_h,n_in)*0.001;
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_fin,n_h)*0.001;
    b2=np.zeros((n_fin,1))
    
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    #print parameters;
    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''
    ### CODE HERE
    # print A.shape
    # print W.shape
    # print b.shape
    #print b.shape
    Arow,Acol=A.shape

    #print b.shape
    Z=np.dot(A.T,W.T) + b.T

    
    Z=Z.T

    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    #print "Zshape=",Z.shape
    #print "Z=",Z
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    
    # print A.shape
    # print A
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    '''
    Estimates the cost with prediction A2
	uses the Logistic Regression Cost function
    Inputs:
        A2 - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels
    
    Returns:
        cost of the objective function: Logistic Regression Cost function
    '''
    ### CODE HERE
    #print A2
    # print Y
    costfn= ( np.multiply(Y,np.log(A2)) + np.multiply((1-Y),np.log(1-A2)));
    cost=-1*np.sum(costfn)
    cost= cost/(Y.size)
    # print cost.size
    # print cost
    return cost

def linear_backward(dZ, cache, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE
    # print "dzshape=",dZ.shape
    # # print cache["A"].shape
    # print "W.shape=",W.shape
    # print "b.shape=",b.shape
    # print "cache[A].shape=",cache["A"].shape

    dW=np.dot(dZ,cache["A"].T)
    # maxval=np.amax(dZ)
    # temp=dZ-maxval
    db=np.sum(dZ,axis=1,keepdims=True)#np.sum(dZ,axis=1,keepdims=True) previously did np.dot(dZ,1)
    #print "db.shape=",db.shape    
    dA_prev=np.dot(W.T,dZ)
    # print dW.shape
    # print db.shape
    # print dA_prev.shape
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)

    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def compute_dA(A,Y):
    #Computes dA
    # print A.shape
    Arow,Acol=Y.shape
    #print "1-A=",1-A
    dA= -1*(np.divide(Y,A)) + np.divide((1-Y),(1-A))
    dA=dA/Acol
    #print dA
    return dA

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    A1,cache1 = layer_forward(X,parameters["W1"],parameters["b1"],activation="sigmoid")
    A2,cache2 = layer_forward(A1,parameters["W2"],parameters["b2"],activation="sigmoid")    
    
    Arow,Acol=A2.T.shape
    A2[A2<0.5]=0;
    A2[A2>=0.5]=1;
    
    # for 2 layers
    # YPred=np.zeros(Arow)
    # print Acol
    # for i in range(0,Arow):
    # 	if A2[0][i]>A2[1][i]:
    # 		YPred[i]=0
    # 	else:
    # 		YPred[i]=1
   
    YPred = A2
    #print YPred.shape

    
    
   # for i in range(0,):
   
    return YPred

def two_layer_network(X, Y, VD,VL, net_dims, num_iterations=200, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    
    A0 = X
    #print X.shape
    costs = []
    Vcosts = []
    for ii in range(num_iterations):
        # Forward propagation
        
        ### CODE HERE
        A1,cache1 = layer_forward(A0,parameters["W1"],parameters["b1"],activation="sigmoid")        
        A2,cache2 =	layer_forward(A1,parameters["W2"],parameters["b2"],activation="sigmoid")
        
        #print A2
        # cost estimation
        ### CODE HERE
        cost= cost_estimate(A2,Y)
        ### For Validation Cost now
        VA1,Vcache1 = layer_forward(VD,parameters["W1"],parameters["b1"],activation="sigmoid")        
        VA2,Vcache2 =	layer_forward(VA1,parameters["W2"],parameters["b2"],activation="sigmoid")
        Vcost= cost_estimate(VA2,VL)
        

        # Backward Propagation
        ### CODE HERE
        dA2= compute_dA(A2,Y)
       # print dA2.shape

        dA1,dW2,db2=layer_backward(dA2,cache2,parameters["W2"],parameters["b2"],activation="sigmoid")
        dA0,dW1,db1=layer_backward(dA1,cache1,parameters["W1"],parameters["b1"],activation="sigmoid")
        
        # print parameters["W2"]
        # print parameters["b2"]
        # print parameters["W1"]
        # print parameters["b1"]
        #update parameters	
       # print "before"
        ### CODE HERE
        parameters["W2"]=parameters["W2"] - (learning_rate)*dW2
        parameters["b2"]=parameters["b2"] - (learning_rate)*db2
        parameters["W1"]=parameters["W1"] - (learning_rate)*dW1
        parameters["b1"]=parameters["b1"] - (learning_rate)*db1

        # print parameters["W2"]
        # print parameters["b2"]

        if ii % 10 == 0:
            costs.append(cost)
            Vcosts.append(Vcost)
        if ii % 10 == 0:
            print("Training Cost at iteration %i is: %f" %(ii, cost))
            print("Validation Cost at iteration %i is: %f" %(ii, Vcost))
    
    return costs,Vcosts,parameters

def calculateAccuracy(pred,label):
    counter=0
    #print pred.size
    for i in range(0,pred.size-1):
	    if pred[0][i]==label[0][i]:
	      counter=counter+1
	    # print "pred",pred[0][i]  
	    # print "label",label[0][i]  
    
    accuracy=counter*(1.0)/pred.size
    return accuracy

def train_validation_split(train_data,train_label):
    tv_data=np.split(train_data.T,[1000,1400])
    tvl_data=np.split(train_label.T,[1000,1400])
    
    train_data=[]
    train_label=[]
    train_data=tv_data[0]
    train_label=tvl_data[0]
    
    train_data=np.append(train_data,tv_data[2],axis=0)
    train_label=np.append(train_label,tvl_data[2],axis=0)
    
    validation_data=tv_data[1].T
    validation_label=tvl_data[1].T

    train_data=train_data.T
    train_label=train_label.T
    return train_data,train_label,validation_data,validation_label

def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 1 and 7
    digit_range = [1,7]
    train_data, train_label, test_data, test_label = \
            mnist(noTrSamples=2400,noTsSamples=2000,\
            digit_range=digit_range,\
            noTrPerClass=1200, noTsPerClass=1000)
    #print "train_data=",train_label.shape
    train_data, train_label, validation_data, validation_label = train_validation_split(train_data,train_label)
    #print "train_data=",train_label
        
    #convert to binary labels
    train_label[train_label==digit_range[0]] = 0
    train_label[train_label==digit_range[1]] = 1
    test_label[test_label==digit_range[0]] = 0
    test_label[test_label==digit_range[1]] = 1
    validation_label[validation_label==digit_range[0]] = 0
    validation_label[validation_label==digit_range[1]] = 1
    
    n_in, m = train_data.shape
    
    n_fin = 1
    n_h = 800
    net_dims = [n_in, n_h, n_fin]
    
    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 2000

    costs,Vcosts,parameters = two_layer_network(train_data, train_label,validation_data,validation_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate)
    
    
    # compute the accuracy for training set and testing set
    
    train_Pred = classify(train_data, parameters)

    #print "train Pred=",train_Pred
    
    test_Pred = classify(test_data, parameters)

    #print "test Pred=",test_Pred
    trAcc = calculateAccuracy(train_Pred,train_label)
    teAcc = calculateAccuracy(test_Pred,test_label)
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    # CODE HERE TO PLOT costs vs iterations

    #train error vs iterations here
    x=[]
    for i in range(0,200):
    	x.append(i*10)
    print x	
    plt.plot(x,costs,'ro')
    plt.plot(x,Vcosts,'b^')
    plt.show()

if __name__ == "__main__":
    main()




