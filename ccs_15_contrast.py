import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
#hyperparameter
batch_size=128
learning_rate=0.01
##draw pictures
def plot(samples):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
	return fig

##test model 
X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])
X_=tf.Variable(tf.zeros((1,784)))


W1=tf.Variable(tf.zeros([784,10]))
b1=tf.Variable(tf.zeros([10]))
def model(x):
	result=tf.matmul(x,W1)+b1
	return result
var_d1=[W1,b1]
Y1=model(X)
loss_2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y1))
optimizer_2=tf.train.GradientDescentOptimizer(0.5).minimize(loss_2,var_list=var_d1)
correction_prediction=tf.equal(tf.argmax(Y1,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

#initialization
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#training
for i in range(1000):
	batch_x,batch_y=mnist.train.next_batch(batch_size)
	_,loss_val=sess.run([optimizer_2,loss_2],feed_dict={X:batch_x,Y:batch_y})
	if i %100==0:
		print("Iter:{}, Loss_value:{}".format(i,loss_val))

#model_inversion_attack
def attack(class_num,round,initial_value,learning_rate):
	vec=np.zeros(10)
	vec[class_num]=1
	vec=tf.stack([vec])
	vec=tf.cast(vec,tf.float32)
	value=model(initial_value)
	loss=tf.reduce_mean(tf.square(vec-tf.nn.softmax(value)))
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list=[initial_value])
	for i in range(round):
		_,loss_val,val=sess.run([train_step,loss,tf.nn.softmax(value)])
		if i%100==0:
			print("Iter:{}, Loss:{}".format(i,loss_val))
	return initial_value
X_=attack(2,10000,X_,0.01)
image=np.reshape(sess.run(X_),(28,28))
plt.figure(figsize=(1,1))
fig=plt.imshow(image,origin='upper',cmap='gray')
plt.savefig('./{}.png'.format("attack_image_test"), bbox_inches='tight')
