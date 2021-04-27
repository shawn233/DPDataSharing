import torch
import torchvision
import numpy as np
import numpy.linalg as LA
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets
import torchvision.transforms as transforms
####autoencoder definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(32*32, 400)
        self.fc2 = nn.Linear(400,256)
    def forward(self, x):
        x=F.sigmoid(self.fc1(x))
        return F.sigmoid(self.fc2(x))
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(256, 400)
        self.fc2 = nn.Linear(400, 32*32)
    def forward(self, x):
    	x=F.sigmoid(self.fc1(x))
        return F.sigmoid(self.fc2(x))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
    	return self.decoder(self.encoder(x))
autoencoder=Autoencoder().cuda()
criterion_auto=nn.MSELoss().cuda()
optimizer_auto=optim.Adam(autoencoder.parameters(),lr=0.01)

####hyper paramters
learning_rate=0.001
download_fraction=1.
upload_fraction=0.1
batch_size=100
Epoch=10
target=9 #need to change
data_num=20000  
####accuracy
def accuracy(label,output):
	_,prediction=torch.max(output.data,1)
	return (prediction==label).sum()
####load data
transform = transforms.Compose([transforms.Scale(32),
                                transforms.ToTensor()])
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('../cifar10_torch/MNIST', train=True, download=True,transform=transform),batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('../cifar10_torch/MNIST', train=False, transform=transform),batch_size=batch_size,shuffle=True)
All_train_data=torch.Tensor([])
All_train_label=torch.LongTensor([])
j=0
for i,data in enumerate(trainloader,0):
	image,label=data
	All_train_label=torch.cat((All_train_label,label),0)
	All_train_data=torch.cat((All_train_data,image),0)
	j=j+1
	if j==data_num/batch_size:
		break
####model definitions
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Conv2d(1,32,5,padding=2)
		self.pool1=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(32,64,5,padding=2)
		self.pool2=nn.MaxPool2d(2,2)
		self.fc1=nn.Linear(64*4*4,1024)
		self.fc2=nn.Linear(1024,11)
		self.tanh=nn.Tanh()
	def forward(self,x):
		x=self.pool1(self.tanh((self.conv1(x))))
		x=self.pool2(self.tanh(self.conv2(x)))
		x=x.view(-1,1024)
		x=self.tanh(self.fc1(x))
		x=F.sigmoid(self.fc2(x))
		return x
class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.conv1=nn.ConvTranspose2d(100,256,4)
		self.batchnorm1=nn.BatchNorm2d(256)
		self.conv2=nn.ConvTranspose2d(256,128,4,2,1)
		self.batchnorm2=nn.BatchNorm2d(128)
		self.conv3=nn.ConvTranspose2d(128,64,4,2,1)
		self.batchnorm3=nn.BatchNorm2d(64)
		self.conv4=nn.ConvTranspose2d(64,1,3,1,1)
		self.tanh=nn.Tanh()
	def weight_init(self,mean,std):
		for m in self._modules:
			normal_init(self._modules[m],mean,std)
	def forward(self,x):
		x=F.relu(self.batchnorm1(self.conv1(x)))
		x=F.relu(self.batchnorm2(self.conv2(x)))
		x=F.relu(self.batchnorm3(self.conv3(x)))
		x=self.tanh(self.conv4(x))
		return x
'''
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Conv2d(1,32,5,padding=2)
		self.pool1=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(32,64,5,padding=2)
		self.pool2=nn.MaxPool2d(2,2)
		self.fc1=nn.Linear(1024,200)
		self.fc2=nn.Linear(200,11)
		self.tanh=nn.Tanh()
	def forward(self,x):
		x=self.pool1(self.tanh((self.conv1(x))))
		x=self.tanh(self.conv2(x))
		x=x.view(-1,1024)
		x=self.tanh(self.fc1(x))
		x=F.sigmoid(self.fc2(x))
		return x
class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.conv1=nn.ConvTranspose2d(100,256,4)
		self.batchnorm1=nn.BatchNorm2d(256)
		self.conv2=nn.ConvTranspose2d(256,128,4,2,1)
		self.batchnorm2=nn.BatchNorm2d(128)
		self.conv3=nn.ConvTranspose2d(128,64,4,2,1)
		self.batchnorm3=nn.BatchNorm2d(64)
		self.conv4=nn.ConvTranspose2d(64,1,3,1,1)
		self.tanh=nn.Tanh()
	def weight_init(self,mean,std):
		for m in self._modules:
			normal_init(self._modules[m],mean,std)
	def forward(self,x):
		x=F.relu(self.batchnorm1(self.conv1(x)))
		x=F.relu(self.batchnorm2(self.conv2(x)))
		x=F.relu(self.batchnorm3(self.conv3(x)))
		x=self.tanh(self.conv4(x))
		return x
'''
def normal_init(m,mean,std):
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()
G=Generator().cuda()
BCE_loss = nn.BCELoss().cuda()
G.weight_init(mean=0.0,std=0.02)
G_optimizer=optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))

server=Net().cuda()
criterion=nn.CrossEntropyLoss().cuda()
optimizer_server=optim.Adam(server.parameters(), lr=learning_rate, betas=(0.5, 0.999))

local_model=Net().cuda()
local_optimizer=optim.Adam(local_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

attacker_model=Net().cuda()
attacker_optimizer=optim.Adam(attacker_model.parameters(),lr=learning_rate,betas=(0.5,0.999))

####split data for every participant
def reform_data(data):
	result=autoencoder.encoder(Variable(data).view(-1,32*32).cuda())
	result=result.data.view(-1,1,16,16)
	#result=np.pad(result,(8,8),'constant',constant_values=0)
	#result=np.reshape(result,(-1,1,32,32))
	return result
def split_data(index):
	num=0
	a=[]
	attacker_data=torch.Tensor([])
	attacker_label=torch.LongTensor([])
	local_data=torch.Tensor([])
	local_label=torch.LongTensor([])
	for i in range(10):
		if i!=index:
			a.append(i)
	a=np.array(a)
	index_attack=np.array([0,1]) #need to change
	index_local=np.array([index,0])#need to change
	for i in range(data_num):
		if All_train_label[i] in index_local:
			if All_train_label[i]==index:
				num+=1
			local_data=torch.cat((All_train_data[i].unsqueeze(0),local_data),0)
			local_label=torch.cat((torch.LongTensor([All_train_label[i]]),local_label),0)
		if All_train_label[i] in index_attack:
			attacker_data=torch.cat((All_train_data[i].unsqueeze(0),attacker_data),0)
			attacker_label=torch.cat((torch.LongTensor([All_train_label[i]]),attacker_label),0)
	print("target images ")+str(index)+str(' ')+str(num)
	return attacker_data,attacker_label,local_data,local_label
####training_autoencoder
running_loss=0.
for epoch in range(20):
	for i, data in enumerate(trainloader,0):
		inputs,labels=data
		inputs,labels=Variable(inputs).view(-1,32*32).cuda(),Variable(labels).cuda()
		optimizer_auto.zero_grad()
		output=autoencoder(inputs)
		loss=criterion_auto(output,inputs)
		loss.backward()
		optimizer_auto.step()
		if i %100==0:
			print("epoch:{}, iter:{}, loss:{}".format(epoch+1,i+1,running_loss/2000.0))
print("finished autoencoder training")
####training_protocol
def get_dimension(shape):
	num=1
	for i in shape:
		num*=i
	return num,shape
def download_parameters():
	return server.state_dict()
def upload_parameters(prev_params,cur_params):
	for k in cur_params:
		cur_params[k]=cur_params[k]-prev_params[k]
	all_parameters=[]
	for k in cur_params:
		val=cur_params[k]
		shape=val.size()
		num,result=get_dimension(shape)
		val=val.view([1,num])
		for j in range(num):
			all_parameters.append(abs(val[0][j]))
	all_parameters=sorted(all_parameters,reverse=True)
	flag=all_parameters[int(len(all_parameters)*upload_fraction)]
	for j in cur_params:
		val=cur_params[j]
		shape=val.size()
		num,result=get_dimension(shape)
		val=val.view([-1,num])
		for k in range(num):
			if abs(val[0][k])<flag:
				val[0][k]=0.
		val=val.view(result)
		cur_params[j]=val
	return cur_params
def fetch(name,index,data,label):
		size=data.size(0)
		leftsize=(index*batch_size)%size
		rightsize=min(leftsize+batch_size,size)
		return data[leftsize:rightsize],label[leftsize:rightsize]
def simulation():
	attacker_data,attacker_label,local_data,local_label=split_data(target)
	attacker_label,local_label=attacker_label.cuda(),local_label.cuda()
	attacker_data=reform_data(attacker_data).cuda()
	local_data=reform_data(local_data).cuda()
	print ("attacker data"),attacker_data.size()
	print ("users data"),local_data.size()
	for e in range(Epoch):
		####training local
		params=download_parameters()
		local_model.load_state_dict(params) 
		data_size=local_data.size(0)
		for k in range(data_size/batch_size):
			#ones=torch.ones([batch_size]).cuda()
			image,label=fetch("local",k,local_data,local_label)
			image,label=Variable(image),Variable(label)
			output=local_model(image)
			local_optimizer.zero_grad()
			loss=criterion(output[:,0:10],label)
			loss.backward()
			local_optimizer.step()
			if k%30==0:
				print ("Epoch:{}, Round:{}, finishes for users".format(e,k))
		result=upload_parameters(params,local_model.state_dict())
		server_params=server.state_dict()
		for k in server.state_dict():
			server_params[k]+=result[k]
		server.load_state_dict(server_params)
		print ("Epoch:{} finishes for users".format(e))
		total=0.0
		correct=0.0
		data_size=local_data.size(0)
		####training attacker
		##poisoing
		attacker_model.load_state_dict(params)
		y_real=Variable(torch.ones(batch_size,1)).cuda()
		z_ = torch.randn((batch_size, 100)).view(-1,100,1,1)
		z_ = Variable(z_).cuda()
		G_optimizer.zero_grad()
		G_result=G(z_)
		print G_result.size()
		D_result=attacker_model(G_result)
		label=torch.LongTensor([]).cuda()
		for j in range(batch_size):
			label=torch.cat((label,torch.LongTensor([target]).cuda()),0)
		#G_train_loss=BCE_loss(D_result[:,10:11],y_real)+criterion(D_result[:,0:10],Variable(label))
		G_train_loss=criterion(D_result[:,0:10],Variable(label))
		print ("Epoch:{} finishes, generator loss:{}".format(e,G_train_loss))
		G_train_loss.backward()
		G_optimizer.step()

		z_ = torch.randn((batch_size, 100)).view(-1,100,1,1).cuda()
		z_ = Variable(z_)
		G_result=G(z_)
		attacker_data=torch.cat((attacker_data,G_result.data),0)
		for j in range(batch_size):
			attacker_label=torch.cat((attacker_label,torch.LongTensor([1]).cuda()),0)#need to change
		##train
		data_size=attacker_data.size(0)
		for k in range(data_size/batch_size):
			image,label=fetch("attacker",k,attacker_data,attacker_label)
			image,label=Variable(image),Variable(label)
			output=attacker_model(image)[:,0:10]
			attacker_optimizer.zero_grad()
			loss=criterion(output,label)
			loss.backward()
			attacker_optimizer.step()
			if k%30==0:
				print ("Epoch:{}, Round:{}, finishes for attackers".format(e,k))
		result=upload_parameters(params,attacker_model.state_dict())
		server_params=server.state_dict()
		for k in server.state_dict():
			server_params[k]+=result[k]
		server.load_state_dict(server_params)
		print ("Epoch:{} finishes for attackers".format(e))
		data_size=attacker_data.size(0)
	for k in range(data_size/batch_size):
		image,label=fetch("local",k,local_data,local_label)
		output=server(Variable(image))
		correct+=accuracy(label,output[:,0:10])
		total+=label.size(0)
		print("Epoch:{}, user Accuracy :{}%".format(e,100. * correct / total))
simulation()
torch.save(G.cpu().state_dict(),'./g_defense_9.pt')
torch.save(server.cpu().state_dict(),'./server_defense_9.pt')

