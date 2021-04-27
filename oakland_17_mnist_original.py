import tensorflow as tf
import numpy as np 
import random
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('./data', one_hot=True)
##variable
X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])
Y_=tf.placeholder(tf.float32,shape=[None,2])
learning_rate=0.01
batch_size=50

##target model
W1=tf.Variable(tf.zeros([256,10]))
b1=tf.Variable(tf.zeros([10]))
def model(x):
	result=tf.matmul(x,W1)+b1
	return result
Y1=model(X)
var_d1=[W1,b1]
loss_2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y1))
optimizer_2=tf.train.GradientDescentOptimizer(0.5).minimize(loss_2,var_list=var_d1)
correction_prediction=tf.equal(tf.argmax(Y1,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correction_prediction,tf.float32))


##attack models and shadow models
def fetch_data(index):
	index=random.sample(range(0, 50000), 1000)
	training_data=[]
	training_label=[]
	for i in index:
		training_data.append(mnist.train.images[i])
		training_label.append(mnist.train.labels[i])
	training_data=np.array(training_data)
	training_label=np.array(training_label)
	return [training_data,training_label]
def fetch_data2(index):
	index=random.sample(range(0, 10000), 1000)
	test_data=[]
	test_label=[]
	for i in index:
		test_data.append(mnist.test.images[i])
		test_label.append(mnist.test.labels[i])
	test_data=np.array(test_data)
	test_label=np.array(test_label)
	return [test_data,test_label]
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

target_data,target_label=fetch_data(0)
target_test_data,target_test_label=fetch_data2(0)
for i in range(100):
	_,loss_val=sess.run([optimizer_2,loss_2],feed_dict={X:target_data,Y:target_label})
	if(i%10==0):
		print ("Iter:{}, loss:{}".format(i,loss_val))
##launch attack

#training shadow models
for j in range(50):
	loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=shadow_models[j])))
	train_optimizer.append(tf.train.GradientDescentOptimizer(learning_rate).minimize(loss[j]))
for i in range(50):
	for j in range(100):
		_,loss_val=sess.run([train_optimizer[i],loss[i]],feed_dict={X:training_data[i],Y:training_label[i]})
		if j % 20==0:
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
	test_data,test_label=fetch_data2(i)
	result=sess.run(shadow_models[i],feed_dict={X:test_data})
	for j in range(result.shape[0]):
		label=np.argmax(result[j])
		tmp=np.reshape(np.array(result[j]),(1,-1))
		training_attack_data[label].append(tmp)
		training_attack_label[label].append([0.,1.])
for i in range(50):
	result=sess.run(shadow_models[i],feed_dict={X:training_data[i]})
	for j in range(result.shape[0]):
		label=np.argmax(result[j])
		tmp=np.reshape(np.array(result[j]),(1,-1))
		training_attack_data[label].append(tmp)
		training_attack_label[label].append([1.,0.])
for i in range(10):
	training_attack_data[i]=np.reshape(np.array(training_attack_data[i]),(-1,10))
	training_attack_label[i]=np.reshape(np.array(training_attack_label[i]),(-1,2))
for i in range(10):
	training_attack_loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=attack_models[i])))
	training_attack_optimizer.append(tf.train.GradientDescentOptimizer(0.5).minimize(training_attack_loss[i]))
#training attack models
for i in range(10):
	for j in range(100):
		_,loss_val=sess.run([training_attack_optimizer[i],training_attack_loss[i]],feed_dict={Y:training_attack_data[i],Y_:training_attack_label[i]})
		if i %10==0:
			print("Attack:{}, Iter:{}, loss_val:{}".format(i,j,loss_val))

#testing attack model
total=0.
correct=0.
kind_total=np.zeros(10)
kind_correct=np.zeros(10)
for i in range(len(target_test_data)):
	data=np.reshape(np.array(target_test_data[i]),(1,784))
	result_y=sess.run(Y1,feed_dict={X:data})
	label=np.argmax(result_y)
	output=sess.run(attack_models[label],feed_dict={Y:np.reshape(result_y,(1,10))})
	result_y=int(np.argmax(output))
	index=int(np.argmax(target_test_label[i]))
	if result_y==0:
		total+=1
		kind_total[index]+=1
for i in range(len(target_test_data)):
	data=np.reshape(np.array(target_data[i]),(1,784))
	result_y=sess.run(Y1,feed_dict={X:data})
	label=int(np.argmax(result_y))
	output=sess.run(attack_models[label],feed_dict={Y:np.reshape(result_y,(1,10))})
	result_y=int(np.argmax(output))
	index=int(np.argmax(target_label[i]))
	if result_y==0:
		correct+=1
		kind_correct[index]+=1
		total+=1
		kind_total[index]+=1
print kind_total
print kind_correct
print correct/total
