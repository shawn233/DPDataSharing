import tensorflow as tf
import numpy as np




def gen_samples(path):
	data=np.genfromtxt(fname=path,delimiter=',',dtype=str,autostrip=True)
	dict={1:{'Private':0,'Self-emp-not-inc':1,'Self-emp-inc':2,'Federal-gov':3,'Local-gov':4,'State-gov':5,'Without-pay':6,'Never-worked':7,'?':-1},
	      3:{'Bachelors':0,'Some-college':1,'11th':2,'HS-grad':3,'Prof-school':4,'Assoc-acdm':5,'Assoc-voc':6,'9th':7,'7th-8th':8,'12th':9,'Masters':10,'1st-4th':11,'10th':12,'Private':13,'Doctorate':14,'5th-6th':15,'Preschool':16,'?':-1},
	      5:{'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3,'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6,'?':-1},
	      6:{'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11,'Protective-serv':12,'Armed-Forces':13,'?':-1},
	      7:{'Wife':0,'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5,'?':-1},
	      8:{'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4,'?':-1},
	      9:{'Female':0, 'Male':1,'?':-1},
	      13:{'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19, 'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24, 'Laos':25, 'Ecuador':25, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40,'?':-1},
	      14:{'>50K':0, '<=50K':1,'>50K.':0, '<=50K.':1}}
	'''
	for i in range(np.shape(data)[0]):
	    for j in range(np.shape(data)[1]):
	        
	        if j in dict.keys():
	            data[i][j]=dict[j][data[i][j]]
	        else: data[i][j]=float(data[i][j])
	data=data.astype(np.float)
	labels=data[:,14]
	
	Y=np.zeros((np.shape(data)[0],2))
	
	for i in range(np.shape(Y)[0]):
		if labels[i] ==1.0:
			Y[i][1]=1.0
		if labels[i] ==0.0:
			Y[i][0]=1.0
	
	
	X=data[:,:-1]/data[:,:-1].max(axis=0)
	'''
	X=np.zeros((np.shape(data)[0],106),dtype="float32")
	Y=np.zeros((np.shape(data)[0],2),dtype="float32")
	for i in range(np.shape(data)[0]):
		offset=0
		for j in range(np.shape(data)[1]-1):
			if j in dict.keys():
				if dict[j][data[i][j]]>=0:
					X[i][offset+dict[j][data[i][j]]]=1
				offset+=(len(dict[j])-1)  ##BE CAREFULL there is still a key"?"
		#	else:
		#		X[i][offset]=np.float32(data[i][j])
		#		offset+=1
		if dict[14][data[i][np.shape(data)[1]-1]]==1.0:
			Y[i][1]=1.0
		else:
			Y[i][0]=1.0



	return X,Y

X_train,Y_train=gen_samples('./adult.data')
X_test,Y_test=gen_samples('./adult.test')
#hyper paramters
learning_rate=0.001
training_epochs=100
batch_size=128
#auto-encoder
X=tf.placeholder(tf.float32,[None,106])
Y=tf.placeholder(tf.float32,[None,2])
weights_au={
	'encoder_h1': tf.Variable(tf.random_normal([106,5])),
	#'encoder_h2': tf.Variable(tf.random_normal([400,256])),
	'decoder_h1': tf.Variable(tf.random_normal([5,106])),
	#'decoder_h2': tf.Variable(tf.random_normal([400,784])),
}
bias_au={
	'encoder_b1':tf.Variable(tf.random_normal([5])),
	#'encoder_b2':tf.Variable(tf.random_normal([256])),
	'decoder_b1':tf.Variable(tf.random_normal([106])),
	#'decoder_b2':tf.Variable(tf.random_normal([784])),
}

def encoder(x):
	layer_1=tf.nn.sigmoid(tf.matmul(x,weights_au['encoder_h1'])+bias_au['encoder_b1'])
	#layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights_au['encoder_h2'])+bias_au['encoder_b2'])
	#return layer_2
	return layer_1

def decode(x):
	layer_1=tf.nn.sigmoid(tf.matmul(x,weights_au['decoder_h1'])+bias_au['decoder_b1'])
	#layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights_au['decoder_h2'])+bias_au['decoder_b2'])
	#return layer_2
	return layer_1

encode_op=encoder(X)
decode_op=decode(encode_op)
loss_1=tf.reduce_mean(tf.square(X-decode_op))
optimizer_1= tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1)



#network_parameters
n_hidden_1=50
n_hidden_2=50
n_input=256
n_classes=10

#multiplayer_perceptron network structure
'''
weights={
	'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes])),
}
bias={
	'b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'out':tf.Variable(tf.random_normal([n_classes])),
}
'''

#varible with autoencoder

weights={
	'h1':tf.Variable(tf.random_normal([5,4])),
	'h2':tf.Variable(tf.random_normal([4,3])),
	'out':tf.Variable(tf.random_normal([3,2])),
}
bias={
	'b1':tf.Variable(tf.random_normal([4])),
	'b2':tf.Variable(tf.random_normal([3])),
	'out':tf.Variable(tf.random_normal([2])),
}

def multilayer_perceptron(x):
	layer_1=tf.nn.relu(tf.matmul(x,weights['h1'])+bias['b1'])
	layer_2=tf.nn.relu(tf.matmul(layer_1,weights['h2'])+bias['b2'])
	layer_3=tf.matmul(layer_2,weights['out'])+bias['out']

	return layer_3


output=multilayer_perceptron(encode_op)
#output=multilayer_perceptron(X)
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
train_step=tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

initial=tf.global_variables_initializer()
sess=tf.Session()


#initial
sess.run(initial)

#training_auto-encoder
for i in range(3000):
	
	_,loss_val=sess.run([optimizer_1,loss_1],feed_dict={X:X_train})
	if(i%100==0):
		print("Iter:{}, loss:{}".format(i,loss_val))
'''
	if(i%100==0):		
		for j in range(n):
			batch_x,_=mnist.test.next_batch(n)
			tmp=sess.run(decode_op,feed_dict={X:batch_x})
		#	for k in range(n):
		#		canvas_orig[j*(28):(j+1)*28,k*(28):(k+1)*28]=batch_x[j].reshape([28,28])
		#	for k in range(n):
			#	canvas_recon[j*(28):(j+1)*28,k*(28):(k+1)*28]=tmp[j].reshape([28,28])
		#print("origin image:")
		#plt.figure(figsize=(n,n))
		#fig=plt.imshow(canvas_orig,origin='upper',cmap='gray')
		#plt.savefig('./picture/{}.png'.format(str(i)+"_orgi"), bbox_inches='tight')
		#plt.close(fig)

		#print("reconstructed image:")
		#plt.figure(figsize=(n,n))
		#fig=plt.imshow(canvas_recon,origin='upper',cmap='gray')
		#plt.savefig('./picture/{}.png'.format(str(i)+"_recon"), bbox_inches='tight')
'''
#training_multiplayer

for i in range(10000):
	
	_,loss_val=sess.run([train_step,cost],feed_dict={X:X_train,Y:Y_train})
	if i%200==0:
		print("Iter:{}, accuracy:{}".format(i,sess.run(accuracy,feed_dict={X:X_test,Y:Y_test})))

print("accuracy:{}".format(sess.run(accuracy,feed_dict={X:X_test,Y:Y_test})))

