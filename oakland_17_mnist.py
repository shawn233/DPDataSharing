import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
##variable
X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])
Y_=tf.placeholder(tf.float32,shape=[None,2])
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


##attack models and shadow models
def fetch_data(index):
	training_data=mnist.train.images[index*1000:(index+1)*1000]
	training_label=mnist.train.labels[index*1000:(index+1)*1000]
	return [training_data,training_label]
def bulild_shadow_models(x):
	W=tf.Variable(tf.zeros([784,10]))
	b=tf.Variable(tf.zeros([10]))
	return tf.matmul(x,W)+b
def bulild_attack_model(x):
	W=tf.Variable(tf.zeros([10,2]))
	b=tf.Variable(tf.zeros([2]))
	return tf.matmul(x,W)+b
def annote_data(data,test):
	if test:
		return np.reshape(np.zeros(data.shape[0]),(-1,1))
	else:
		return np.reshape(np.ones(data.shape[0],(-1,1)))
shadow_models=[]
attack_models=[]
training_data=[]
training_label=[]
train_optimizer=[]
loss=[]
for i in range(50):
	shadow_models.append(bulild_shadow_models(X))
	train,label=fetch_data(i)
	training_data.append(train)
	training_label.append(label)
for j in range(10):
	attack_models.append(bulild_attack_model(Y))

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

#training shadow models
for j in range(50):
	loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=shadow_models[j])))
	train_optimizer.append(tf.train.GradientDescentOptimizer(learning_rate).minimize(loss[j]))
for i in range(50):
	for j in range(100):
		_,loss_val=sess.run([train_optimizer[i],loss[i]],feed_dict={X:training_data[i],Y:training_label[i]})
		if j % 50==0 and j>0:
			print("Shadow:{}, Iter:{}, Loss:{}".format(i,j,loss_val))

#attack models
training_attack_data=[]
training_attack_label=[]
training_attack_loss=[]
training_attack_optimizer=[]
for i in range(10):
	training_attack_data.append([])
	training_attack_label.append([])
for i in range(50):
	result=sess.run(shadow_models[i],feed_dict={X:mnist.test.images[i*100:(i+1)*100]})
	for j in range(result.shape[0]):
		label=np.argmax(result[j])
		tmp=np.reshape(np.array(result[j]),(1,-1))
		training_attack_data[label].append(tmp)
		training_attack_label[label].append([0,1])
for i in range(50):
	result=sess.run(shadow_models[i],feed_dict={X:training_data[i]})
	for j in range(result.shape[0]):
		label=np.argmax(result[j])
		tmp=np.reshape(np.array(result[j]),(1,-1))
		training_attack_data[label].append(tmp)
		training_attack_label[label].append([1,0])
for i in range(10):
	training_attack_data[i]=np.reshape(np.array(training_attack_data[i]),(-1,10))
	training_attack_label[i]=np.reshape(np.array(training_attack_label[i]),(-1,2))
for i in range(10):
	training_attack_loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=attack_models[i])))
	training_attack_optimizer.append(tf.train.GradientDescentOptimizer(learning_rate).minimize(training_attack_loss[i]))
#training attack models
for i in range(10):
	for j in range(100):
		_,loss_val=sess.run([training_attack_optimizer[i],training_attack_loss[i]],feed_dict={Y:training_attack_data[i],Y_:training_attack_label[i]})
		if i %50==0:
			print("Attack:{}, Iter:{}, loss_val:{}".format(i,j,loss_val))

#testing attack model
total=0.
correct=0.
for i in range(5000):
	data=np.reshape(np.array(mnist.test.images[i]),(1,784))
	result_y=sess.run(Y1,feed_dict={X:data})
	label=np.argmax(result_y)
	output=sess.run(attack_models[label],feed_dict={Y:np.reshape(result_y,(1,10))})
	result_y=np.argmax(output)
	if result_y:
		correct+=1
	total+=1
for i in range(10000):
	data=np.reshape(np.array(mnist.train.images[i]),(1,784))
	result_y=sess.run(Y1,feed_dict={X:data})
	label=np.argmax(result_y)
	output=sess.run(attack_models[label],feed_dict={Y:np.reshape(result_y,(1,10))})
	result_y=np.argmax(output)
	if result_y==0:
		correct+=1
	total+=1
print correct/total
