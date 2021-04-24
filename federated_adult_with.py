import torch
import torchvision
import numpy as np
import numpy.linalg as LA
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import time
####hyper paramters
learning_rate=0.001
download_fraction=1.
upload_fraction=0.1
batch_size=100
Epoch=500
data_num=32560  
n=10 #num of users
####load
a1=data=np.genfromtxt(fname='../data_from_chong_xiang_mlaas/dp_data/adult/a1.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1]
b1=data=np.genfromtxt(fname='../data_from_chong_xiang_mlaas/dp_data/adult/b1.matrix',delimiter='\t',dtype=float,autostrip=True)[:,:-1].reshape(6)
def encoded(data):
	data=data.cpu().numpy()
	data=np.reshape(data,(-1,100))
	data=np.dot(data,a1)+b1
	data=np.reshape(data,(-1,6))
	return torch.FloatTensor(data)
####accuracy
def accuracy(label,output):
	_,prediction=torch.max(output.data,1)
	return (prediction==label).sum()
####load data
dataset= pd.read_csv('../data_from_chong_xiang_mlaas/dp_data/adult/adult_training_ori.csv')
number=len(dataset)
features=dataset.ix[:,1:].as_matrix()
targets=dataset.ix[:,0].as_matrix().astype(np.int64)
features=torch.FloatTensor(features)
targets=torch.LongTensor(targets)
adult_dataset = data_utils.TensorDataset(features, targets)
'''
class adultdataset(Dataset):
	def __init__(self,csv_file,root_dir=None,transform=None):
		self.data=pd.read_csv(csv_file)
		self.transform=transform
		self.root_dir=root_dir
	def __len__(self):
		return len(self.data)
	def __getitem__(self,idx):
		return torch.FloatTensor(self.data.ix[idx,1:].as_matrix()),torch.LongTensor(self.data.ix[idx,0].as_matrix())
adult_dataset = adultdataset(csv_file='../data_from_chong_xiang_mlaas/dp_data/adult/adult_training_ori.csv')
'''
trainloader = DataLoader(adult_dataset, batch_size=batch_size,shuffle=True, num_workers=4)		
testloader = DataLoader(adult_dataset, batch_size=batch_size,shuffle=True, num_workers=4)	
def get_all_training_data():
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
	return All_train_data,All_train_label
####model definitions
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.fully=nn.Linear(6,2)
	def forward(self,x):
		x=self.fully(x)
		return x
server=Net()
criterion=nn.CrossEntropyLoss()
optimizer_server=optim.Adam(server.parameters(), lr=learning_rate, betas=(0.5, 0.999))

local_model=[]
local_optimizer=[]
for i in range(n):
	local_model.append(Net())
	local_optimizer.append(optim.Adam(local_model[i].parameters(), lr=learning_rate, betas=(0.5, 0.999)))

def get_data(index,step,data,label):
	size=data_num/n
	leftsize=index*size+step*batch_size
	rightsize=min(leftsize+batch_size,data_num)
	return data[leftsize:rightsize],label[leftsize:rightsize]
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
def test():
	correct=0.
	total=0.
	for i , data in enumerate(testloader,0):
		images,labels=data
		images=encoded(images)
		images=Variable(images)
		outputs=server(images)
		_,prediction=torch.max(outputs.data,1)
		total+=labels.size(0)
		correct+=(prediction==labels).sum()
	print('Accuracy of the network on the 10000 test images:{}'.format(correct / total))
def simulation():
	All_train_data,All_train_label=get_all_training_data()
	for e in range(Epoch):
		####training local
		params=download_parameters()
		for j in range(n):
			local_model[j].load_state_dict(params) 
			data_size=data_num/n
			for k in range(data_size/batch_size):
				image,label=get_data(j,k,All_train_data,All_train_label)
				image=encoded(image)
				image,label=Variable(image),Variable(label)
				output=local_model[j](image)
				local_optimizer[j].zero_grad()
				loss=criterion(output,label)
				loss.backward()
				local_optimizer[j].step()
			result=upload_parameters(params,local_model[j].state_dict())
			server_params=server.state_dict()
			for k in server.state_dict():
				server_params[k]+=result[k]
			server.load_state_dict(server_params)
		if e % 5==0:
			test()
		print ("Epoch:{} finishes for user:{}".format(e,j))
simulation()
