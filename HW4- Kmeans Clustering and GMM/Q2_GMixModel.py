import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import random
from math import pow
import scipy.stats
datasets_dir = '/home/kunal/Desktop/HW4_CSE569/Dataset_1.txt'

def main():
	x=[]
	y=[]	
	with open('Dataset_1.txt') as f:
		for line in f:
			data = line.split()
			x.append(float(data[0]))
			y.append(float(data[1]))

	x=np.asarray(x)
	y=np.asarray(y)		
	print x.shape
	print y.shape

	K=2
	iterations=20
	applyEM(K,x,y,iterations)

def applyEM(K,x,y,iterations,u_k,cov_k):

	rand_k=random.sample(np.arange(0,1600,1),K)
	print rand_k
	
	x_points = np.array(list(zip(x, y)), dtype=np.float32)
	print x_points.shape

	#randomly initializing u_k,cov_k and pi_k

	#u_k=x_points[rand_k]
	# u_k=np.matrix([[ 3.0365329,0.04568216],
 #          [-0.11104848,-0.05193072]])
	# u_k=np.asarray(u_k)
	print "u_kshape=",u_k.shape

	cov=np.cov(x_points.T)*2
	
	cov_k=[]
	for i in range(0,K):
		cov_k.append(cov)

	cov_k=np.asarray(cov_k)	
	
	pi_k=[]
	sum=0
	for i in range(0,K):
		pi=random.uniform(0,0.5)		
		if i == K-1:
			pi=1-sum
		
		sum=sum + pi	
		pi_k.append(pi)

	pi_k=np.asarray(pi_k)	
	
	log=[]
	itr=[]	
	gamma_nk=np.zeros((x_points.shape[0],K))

	for i in range(0,iterations):
		gamma_nk=E(x_points,u_k,cov_k,pi_k)

		u_k,cov_k,pi_k=M(x_points,u_k,cov_k,pi_k,gamma_nk)
		lik=LogLikelihood(x_points,u_k,cov_k,pi_k)
		print lik,"for iteration",i 
		print "u_k=",u_k
		itr.append(i+1)
		log.append(lik)

	print log
	plt.plot(itr,log)
	plt.show()
	color=['red','blue','green','pink','purple','black']

	classes=np.zeros((x_points.shape[0],1))
	for i in range(0,x_points.shape[0]):
		classes[i]=np.argmax(gamma_nk[i])
		print classes[i]

	print x
	print y	
	for i in range(0,x_points.shape[0]):
		plt.scatter(x[i],y[i],c=color[int(classes[i][0])])

	for i in range(0,K):
		plt.scatter(u_k[i][0],u_k[i][1],marker='*',color='black')	
	plt.show()		

def LogLikelihood(x_points,u_k,cov_k,pi_k):
	likelihood=0
	for i in range(0,x_points.shape[0]):
		sumval=0
		for j in range(0,u_k.shape[0]):
			sumval=sumval+pi_k[j]*scipy.stats.multivariate_normal(u_k[j],cov_k[j]).pdf(x_points[i])
		likelihood=likelihood+np.log(sumval)
	return likelihood		

def E(x_points,u_k,cov_k,pi_k):
	
	gamma_nk=np.zeros((x_points.shape[0],u_k.shape[0]))
	for i in range(0,gamma_nk.shape[0]):
		denominator=0
		for j in range(0,gamma_nk.shape[1]):
			denominator=denominator+scipy.stats.multivariate_normal(u_k[j],cov_k[j]).pdf(x_points[i])

		for j in range(0,gamma_nk.shape[1]):
			numerator = scipy.stats.multivariate_normal(u_k[j],cov_k[j]).pdf(x_points[i])
			gamma_nk[i][j]= numerator/denominator

			
	return gamma_nk


def M(x_points,u_k,cov_k,pi_k,gamma_nk):
	
	N_k= np.sum(gamma_nk,axis=0)
	
	for i in range(0,u_k.shape[0]):
		numerator1=0
		numerator2=np.zeros((cov_k.shape[1],cov_k.shape[2]))
		
		
		for j in range(0,x_points.shape[0]):
			numerator1=numerator1+gamma_nk[j][i]*x_points[j]			
		
		u_k[i]=numerator1/N_k[i]

		

		for j in range(0,x_points.shape[0]):			
			numerator2=numerator2+gamma_nk[j][i]*np.dot((x_points[j]-u_k[i]).T.reshape(2,1),(x_points[j]-u_k[i]).reshape(1,2))
		
		cov_k[i]=numerator2/N_k[i]						

	pi_k= N_k/x_points.shape[0]			

	return u_k,cov_k,pi_k	

if __name__ == "__main__":
    main()	