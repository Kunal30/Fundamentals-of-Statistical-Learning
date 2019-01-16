import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import random
from math import pow
from Q2_GMixModel import applyEM

datasets_dir = '/home/kunal/Desktop/HW4_CSE569/Dataset_2.txt'

def main():

	x=[]
	y=[]	
	with open('Dataset_2.txt') as f:
		for line in f:
			data = line.split()
			x.append(float(data[0]))
			y.append(float(data[1]))

	x=np.asarray(x)
	y=np.asarray(y)		
	print x.shape
	print y.shape
	r=5
	K=3
	u_k,r_nk=apply_Kmeans(x,y,r,K)
	
	print "rnk=",r_nk.shape
	maxi=np.argmax(r_nk,axis=1)
	
	x_points = np.array(list(zip(x, y)), dtype=np.float32)
	print maxi.shape
	
	cluster_points0=[]
	cluster_points1=[]
	cluster_points2=[]
	
	for i in range(0,1500):
		if maxi[i]==0:
			cluster_points0.append(x_points[i])
		elif maxi[i]==1:
			cluster_points1.append(x_points[i])
		else:
			cluster_points2.append(x_points[i])	
	
	cluster_points0=np.asarray(cluster_points0)
	cluster_points1=np.asarray(cluster_points1)
	cluster_points2=np.asarray(cluster_points2)
	
	print "0=",cluster_points0
	print "1=",cluster_points1
	print "2=",cluster_points2

	cov_k=np.zeros((3,2,2))
	cov_k[0]=np.cov(cluster_points0.T)
	cov_k[1]=np.cov(cluster_points1.T)
	cov_k[2]=np.cov(cluster_points2.T)
	
	applyEM(3,x,y,20,u_k,cov_k)

	
	color=['red','blue','green','pink','purple','black']
	# cluster_data=[[],[]]
	for i in range(0,1500):
		for j in range(0,K): 			
			if r_nk[i][j]==1:
				plt.scatter(x[i],y[i],c=color[j])
			plt.scatter(u_k[j][0],u_k[j][1],marker='*',color='black')	
	plt.show()

# Euclidean Distance 
def distance(a, b):
    return np.linalg.norm(a - b)

def apply_Kmeans(x,y,r,K):
	rand_k=random.sample(np.arange(0,1500,1),K)
	print rand_k	

	
	points = np.array(list(zip(x, y)), dtype=np.float32)
	print points.shape
	
	u_k=points[rand_k]
	print u_k
	
	SSE=[]
	indices=[]
	r_nk=np.zeros((1600,K))
	
	for index in range(0,r):
		
		r_nk=np.zeros((1500,K))
		sse=0
		for i in range(0,1500):
			mind=9999999
			min_index=0
			for j in range(0,K):
				d1=distance(points[i],u_k[j])
				if d1 < mind:
					mind=d1
					min_index=j
			r_nk[i][min_index]=1
			sse=sse+pow(mind,2)
		SSE.append(sse)
		ones=np.count_nonzero(r_nk,axis=0)
		
		u_k= np.dot(r_nk.T,points)
		
		for i in range(0,K):
			u_k[i]=u_k[i]/ones[i];
		
		print ones
		plt.plot(sse/1500,index)
		plt.show()
		print "uk=",u_k
		print "SSE=",sse/1500

		indices.append(index+1)

		print r_nk

	print indices
	SSE=np.asarray(SSE)
	SSE=SSE/1500
	plt.plot(indices,SSE)
	plt.show()

	return u_k,r_nk	

	
if __name__ == "__main__":
    main()