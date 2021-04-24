import tensorflow as tf
import numpy as np 
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy.linalg as LA
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
	X=np.zeros((np.shape(data)[0],102),dtype="float32")
	Y=np.zeros((np.shape(data)[0],2),dtype="float32")
	for i in range(np.shape(data)[0]):
		offset=0
		for j in range(np.shape(data)[1]-1):
			if j in dict.keys():
				if dict[j][data[i][j]]>=0:
					X[i][offset+dict[j][data[i][j]]]=1
				offset+=(len(dict[j])-1)  ##BE CAREFULL there is still a key"?"
			else:
				if(j==0 or j==12):
					X[i][offset]=np.float32(data[i][j])
					offset+=1
		if dict[14][data[i][np.shape(data)[1]-1]]==1.0:
			Y[i][1]=1.0
		else:
			Y[i][0]=1.0



	return X,Y

X_train,Y_train=gen_samples('./adult.data')
X_test,Y_test=gen_samples('./adult.test')
for i in range(40):
	print np.sum(X_test,axis=1)
#variable
#X_train=preprocessing.scale(X_train)
#X_test=preprocessing.scale(X_test)
X=tf.placeholder(tf.float32,shape=[None,102])
Y=tf.placeholder(tf.float32,shape=[None,2])
learning_rate=0.5
batch_size=50

#auto-encoder
weights={
	'encoder_h1': tf.Variable(tf.random_normal([102,50])),
	#'encoder_h2': tf.Variable(tf.random_normal([400,256])),
	'decoder_h1': tf.Variable(tf.random_normal([50,102])),
	#'decoder_h2': tf.Variable(tf.random_normal([400,784])),
}
bias={
	'encoder_b1':tf.Variable(tf.random_normal([50])),
	#'encoder_b2':tf.Variable(tf.random_normal([256])),
	'decoder_b1':tf.Variable(tf.random_normal([102])),
	#'decoder_b2':tf.Variable(tf.random_normal([784])),
}

def encoder(x):
	layer_1=tf.nn.sigmoid(tf.matmul(x,weights['encoder_h1'])+bias['encoder_b1'])
	#layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['encoder_h2'])+bias['encoder_b2'])
	return layer_1

def decode(x):
	layer_1=tf.matmul(x,weights['decoder_h1'])+bias['decoder_b1']
	#layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['decoder_h2'])+bias['decoder_b2'])
	return layer_1
encode_op=encoder(X)

decode_op=decode(encode_op)

loss_1=tf.reduce_mean(tf.square(X-decode_op))
optimizer_1= tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1)

#test model
W1=tf.Variable(tf.zeros([50,2]))
b1=tf.Variable(tf.zeros([2]))
Y1=tf.matmul(encode_op,W1)+b1
#Y1=tf.matmul(X,W1)+b1
var_d1=[W1,b1]
loss_2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y1))
optimizer_2=tf.train.GradientDescentOptimizer(0.01).minimize(loss_2,var_list=var_d1)
correction_prediction=tf.equal(tf.argmax(Y1,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

#initialization
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#saver = tf.train.Saver()
#saver.restore(sess, "./one_layer.ckpt")

for i in range(50):
	
	_,loss_val=sess.run([optimizer_1,loss_1],feed_dict={X:X_train})
	if(i%50==0):
		print("Iter:{}, loss:{}".format(i,loss_val))
#saver_path = saver.save(sess, "./one_layer.ckpt")
#print "Model saved in file: ", saver_path
'''
for i in range(10000):
	
	_,loss_val=sess.run([optimizer_2,loss_2],feed_dict={X:X_train,Y:Y_train})
	if(i%200==0):
		print ("Iter:{}, loss:{}".format(i,loss_val))
		print(sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
'''
def generate_data():
	f=open("generated_data_2",'w')
	for i in range(len(X_train)):
		x=np.array([X_train[i]])
		tmp=sess.run(decode_op,feed_dict={X:x})
		f.write(str(tmp[0])+'\n')
	f.close()
	return
generate_data()
#print("Test accuracy:")
#print(sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
#saver_path = saver.save(sess, "./one_layer.ckpt")
#print "Model saved in file: ", saver_path

def compute_norm(x,p):
	return LA.norm(x,p)


def cdf(p):
	data=[]
	for i in range(len(X_train)):
		x=np.array([X_train[i]])
		tmp=sess.run(encode_op,feed_dict={X:x})
		x=np.reshape(x,[1,100])
		#print np.shape(tmp)
		tmp=np.reshape(tmp,[1,6])
		#print np.shape(tmp)
		tmp=np.pad(tmp,(47),'constant',constant_values=0)
		#print "x",np.shape(x)
		#print np.shape(tmp)
		#print "x",x
		#print "tmp",tmp
		#print "x-tmp",x-tmp
		#print compute_norm(x-tmp,p)
		#print np.sum(abs(x-tmp))
		
		
		if p==0:
			fenzi=30
			fenmu=100
			fenzi+=np.sum(compute_norm((x-tmp)[0],0))
			data.append(fenzi/fenmu*100)
		else:
			if p==2:
				if compute_norm(x,p)==0:
					continue
				data.append((compute_norm(x-tmp,p)+np.random.normal(0, 1, 1)[0])/compute_norm(x,p)*100/8.)
			else:
				if compute_norm(x,p)==0:
					continue
				data.append((compute_norm(x-tmp,p)+np.random.normal(0, 1, 1)[0])/compute_norm(x,p)*100)
		#print np.shape(x-tmp)
		#print x-tmp
	#print data
	data_size=len(data)
	#set bins edges
	data_set=sorted(set(data))
	bins=np.append(data_set, data_set[-1]+1)
	counts, bin_edges = np.histogram(data, bins=bins, density=False)
	counts=counts.astype(float)/data_size
	cdf = np.cumsum(counts)
	return bin_edges[0:-1],cdf
'''
x1,y1=cdf(0)
print len(x1)
x2,y2=cdf(2)
print len(x2)
x3,y3=cdf(np.inf)
print len(x3)
tfile=open('norm_distance_0_2.txt','w')
for item in x1:
	tfile.write("%s " % item)
tfile.write('\n')
for item in y1:
	tfile.write("%s " % item)
tfile.write('\n')
for item in x2:
	tfile.write("%s " % item)
tfile.write('\n')
for item in y2:
	tfile.write("%s " % item)
tfile.write('\n')
for item in x3:
	tfile.write("%s " % item)
tfile.write('\n')
for item in y3:
	tfile.write("%s " % item)
tfile.write('\n')
tfile.close()
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x1, y1, label='L0 norm')
ax.plot(x2, y2, label='L2 norm')
ax.plot(x3, y3, label='L-infinity norm')
plt.ylim((0,1))
plt.xlim((0,200))
ax.grid(linestyle=':')
ax.set_ylabel("CDF")
ax.set_xlabel('Data norm distance(%)')
ax.set_title("Norm distance, Adult")
ax.legend()
plt.savefig('norm_distance.eps', format='eps', dpi=1000)


sess.close()
'''