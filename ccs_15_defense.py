import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
##variable
X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])
X_=tf.Variable(tf.zeros([1,256]))
learning_rate=0.01
batch_size=50
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
##auto-encoder
weights={
	'encoder_h1': tf.Variable(tf.random_normal([784,400])),
	'encoder_h2': tf.Variable(tf.random_normal([400,256])),
	'decoder_h1': tf.Variable(tf.random_normal([256,400])),
	'decoder_h2': tf.Variable(tf.random_normal([400,784])),
}
bias={
	'encoder_b1':tf.Variable(tf.random_normal([400])),
	'encoder_b2':tf.Variable(tf.random_normal([256])),
	'decoder_b1':tf.Variable(tf.random_normal([400])),
	'decoder_b2':tf.Variable(tf.random_normal([784])),
}

def encoder(x):
	layer_1=tf.nn.sigmoid(tf.matmul(x,weights['encoder_h1'])+bias['encoder_b1'])
	layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['encoder_h2'])+bias['encoder_b2'])
	return layer_2

def decode(x):
	layer_1=tf.nn.sigmoid(tf.matmul(x,weights['decoder_h1'])+bias['decoder_b1'])
	layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['decoder_h2'])+bias['decoder_b2'])
	return layer_2
encode_op=encoder(X)
decode_op=decode(encode_op)

loss_1=tf.reduce_mean(tf.square(X-decode_op))
optimizer_1= tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1)

##test model
W1=tf.Variable(tf.zeros([256,10]))
b1=tf.Variable(tf.zeros([10]))
def model(x):
	result=tf.matmul(x,W1)+b1
	return result
Y1=model(encode_op)
var_d1=[W1,b1]
loss_2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y1))
optimizer_2=tf.train.GradientDescentOptimizer(0.5).minimize(loss_2,var_list=var_d1)
correction_prediction=tf.equal(tf.argmax(Y1,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

##initialization
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(30000):
	batch_x,_=mnist.train.next_batch(50)
	_,loss_val=sess.run([optimizer_1,loss_1],feed_dict={X:batch_x})
	if(i%100==0):
		print("Iter:{}, loss:{}".format(i,loss_val))
	n=4
	canvas_orig=np.empty((28*n,28*n))
	canvas_recon=np.empty((28*n,28*n))

	if(i%100==0):		
		for j in range(n):
			batch_x,_=mnist.test.next_batch(n)
			tmp=sess.run(decode_op,feed_dict={X:batch_x})
			for k in range(n):
				canvas_orig[j*(28):(j+1)*28,k*(28):(k+1)*28]=batch_x[j].reshape([28,28])
			for k in range(n):
				canvas_recon[j*(28):(j+1)*28,k*(28):(k+1)*28]=tmp[j].reshape([28,28])
for i in range(1000):
	batch_x,batch_y=mnist.train.next_batch(batch_size)
	_,loss_val=sess.run([optimizer_2,loss_2],feed_dict={X:batch_x,Y:batch_y})
	if(i%100==0):
		print ("Iter:{}, loss:{}".format(i,loss_val))
		print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))

print("Test accuracy:")
print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))

##launch attack
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
			print("Iter:{}, Loss:{}, value:{}".format(i,loss_val,val))
	return initial_value
X_=attack(0,10000,X_,0.01)
image=np.reshape(sess.run(X_),(16,16))
plt.figure(figsize=(1,1))
fig=plt.imshow(image,origin='upper',cmap='gray')
plt.savefig('./{}.png'.format("attack_image"), bbox_inches='tight')

decode_X=np.reshape(sess.run(decode(X_)),(28,28))
fig=plt.imshow(image,origin='upper',cmap='gray')
plt.savefig('./{}.png'.format("decoded_image"), bbox_inches='tight')
