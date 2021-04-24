import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#dataset
mnist = input_data.read_data_sets('../../../MNIST_data', one_hot=True)

#parameters
a1=data=np.genfromtxt(fname='./250/a1.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1]
b1=data=np.genfromtxt(fname='./250/b1.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1].reshape(400)

a2=data=np.genfromtxt(fname='./250/a2.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1]
b2=data=np.genfromtxt(fname='./250/b2.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1].reshape(256)

a3=data=np.genfromtxt(fname='./250/a3.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1]
b3=data=np.genfromtxt(fname='./250/b3.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1].reshape(400)

a4=data=np.genfromtxt(fname='./250/a4.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1]
b4=data=np.genfromtxt(fname='./250/b4.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1].reshape(784)

#experiment
number=len(mnist.train.images)
n=1
for i in range(10000):
	canvas_orig=np.empty((28*n,28*n))
	canvas_recon=np.empty((28*n,28*n))
	canvas_encod=np.empty((16*n,16*n))
	if i%1000==0:
		for j in range(n):
			batch_x,_=mnist.test.next_batch(n)
			new_batch_x=np.dot(batch_x,a1)+b1
			new_batch_x=np.dot(new_batch_x,a2)+b2
			for k in range(n):
				canvas_encod[j*(16):(j+1)*16,k*(16):(k+1)*16]=new_batch_x[j].reshape([16,16])
			new_batch_x=np.dot(new_batch_x,a3)+b3
			new_batch_x=np.dot(new_batch_x,a4)+b4
			for k in range(n):
				canvas_orig[j*(28):(j+1)*28,k*(28):(k+1)*28]=batch_x[j].reshape([28,28])
			for k in range(n):
				canvas_recon[j*(28):(j+1)*28,k*(28):(k+1)*28]=new_batch_x[j].reshape([28,28])
		print("origin image:")
		plt.figure(figsize=(n,n))
		fig=plt.imshow(canvas_orig,origin='upper',cmap='gray')
		plt.savefig('./picture/{}.png'.format(str(i)+"_orgi"), bbox_inches='tight')
		print("reconstruction image:")
		plt.figure(figsize=(n,n))
		fig=plt.imshow(canvas_recon,origin='upper',cmap='gray')
		plt.savefig('./picture/{}.png'.format(str(i)+"_reconstruction"), bbox_inches='tight')
		print("encoded image:")
		plt.figure(figsize=(n,n))
		fig=plt.imshow(canvas_encod,origin='upper',cmap='gray')
		plt.savefig('./picture/{}.png'.format(str(i)+"_encoded"), bbox_inches='tight')
