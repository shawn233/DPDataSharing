import torch
import torchvision
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
from auto_cifar import autoencoder
from torchvision.utils import save_image
import torchvision.transforms as transforms

##load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=0)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
All_train_data=torch.Tensor([])
All_train_label=torch.LongTensor([])
All_test_data=torch.Tensor([])
All_test_label=torch.LongTensor([])
j=0
for _,data in enumerate(trainloader,0):
	image,label=data
	All_train_label=torch.cat((All_train_label,label),0)
	All_train_data=torch.cat((All_train_data,image),0)
	j=j+1
	if j==500:
		break
j=0
for _,data in enumerate(testloader,0):
	image,label=data
	All_test_label=torch.cat((All_test_label,label),0)
	All_test_data=torch.cat((All_test_data,image),0)
	j=j+1
	if j==100:
		break
print All_train_data.size()
print All_test_data.size()
##functions to show an image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x

batch_size=100
set_size=2500
##training data and test data
target_train_index=random.sample(range(1, 50000), set_size)
target_test_index=random.sample(range(1, 10000), set_size)
shadow_train_index=[]
shadow_test_index=[]
for i in range(100):
	shadow_train_index.append(random.sample(range(1, 50000), set_size))
	shadow_test_index.append(random.sample(range(1, 10000), set_size))
##fetch training data or test data
def fetch_train_data(index,i,size):
	leftsize=(i*size)%set_size
	rightsize=min(set_size,leftsize+size)
	return torch.FloatTensor(All_train_data.numpy()[index[leftsize:rightsize]]), torch.LongTensor(All_train_label.numpy()[index[leftsize:rightsize]])
def fetch_test_data(index,i,size):
	leftsize=(i*size)%set_size
	rightsize=min(set_size,leftsize+size)
	return torch.FloatTensor(All_test_data.numpy()[index[leftsize:rightsize]]), torch.LongTensor(All_test_label.numpy()[index[leftsize:rightsize]])
##model definition
#shadow network and attack models
class Shadow(nn.Module):
	def __init__(self):
		super(Shadow,self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
class Attack(nn.Module):
	def __init__(self):
		super(Attack,self).__init__()
		self.fc1=nn.Linear(10,2)
	def forward(self,x):
		x=self.fc1(x)
		return x
Shadow_models=[]
Attack_models=[]
attack_criterion=nn.CrossEntropyLoss().cuda()
Shadow_optimizer=[]
Attack_optimizer=[]
for i in range(100):
	Shadow_models.append(Shadow().cuda())
for i in range(10):
	Attack_models.append(Attack().cuda())
for i in range(100):
	Shadow_optimizer.append(optim.SGD(Shadow_models[i].parameters(),lr=0.001,momentum=0.9))
for i in range(10):
	Attack_optimizer.append(optim.SGD(Attack_models[i].parameters(),lr=0.001,momentum=0.9))
#neural network(NN)
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
net=Net().cuda()
criterion=nn.CrossEntropyLoss().cuda()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

##training cnn
for epoch in range(100):
	for i in range(set_size/batch_size):
		inputs,labels=fetch_train_data(target_train_index,i,batch_size)
		inputs,labels=Variable(inputs).cuda(),Variable(labels).cuda()
		optimizer.zero_grad()
		output=net(inputs)
		loss=criterion(output,labels)
		loss.backward()
		optimizer.step()
		if i% 100==0:
			print("epoch:{}, iter:{}, loss:{}".format(epoch,i,loss.data[0]))
print("finished classifier training")

##training Shadow models
for epoch in range(100):
	for i in range(100):
		for j in range(set_size/batch_size):
			image,label=fetch_train_data(shadow_train_index[i],j,batch_size)
			image=Variable(image)
			label=Variable(label)
			Shadow_optimizer[i].zero_grad()
			output=Shadow_models[i](image.cuda())
			loss=attack_criterion(output,label.cuda())
			loss.backward()
			Shadow_optimizer[i].step()
		print("epoch:{}, shawdow_model:{}".format(epoch,i))
print("Training shadow models finished!")

##training Attack models
Attack_training_data=[]
Attack_training_label=[]
for i in range(10):
	Attack_training_label.append(torch.LongTensor([]).cuda())
	Attack_training_data.append(torch.Tensor([]).cuda())
for i in range(100):
	for j in range(set_size/batch_size):
		image,labels=fetch_train_data(shadow_train_index[i],j,batch_size)
		image=Variable(image)
		output=Shadow_models[i](image.cuda())
		for j in range(batch_size):
			label=labels[j]
			data=output.data[j].resize_(1,10)
			Attack_training_data[label]=torch.cat((Attack_training_data[label],data),0)
			Attack_training_label[label]=torch.cat((Attack_training_label[label],torch.LongTensor([1]).cuda()),0)
for i in range(100):
	for j in range(set_size/batch_size):
		image,labels=fetch_test_data(shadow_test_index[i],j,batch_size)
		image=Variable(image)
		output=Shadow_models[i](image.cuda())
		for j in range(batch_size):
			label=labels[j]
			data=output.data[j].resize_(1,10)
			Attack_training_data[label]=torch.cat((Attack_training_data[label],data),0)
			Attack_training_label[label]=torch.cat((Attack_training_label[label],torch.LongTensor([0]).cuda()),0)
for epoch in range(50):
	for i in range(10):
		size=Attack_training_data[i].size()[0]
		for j in range(size/batch_size):
			data=Attack_training_data[i][j*batch_size:(j+1)*batch_size]
			label=Attack_training_label[i][j*batch_size:(j+1)*batch_size]
			data=Variable(data)
			label=Variable(label)
			Attack_optimizer[i].zero_grad()
			output=Attack_models[i](data.cuda())
			loss=attack_criterion(output,label.cuda())
			loss.backward()
			Attack_optimizer[i].step()
		print("epoch:{}, Attack model:{} finishes".format(epoch,i))
#making predictions
correct=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
total=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
for i in range(0,set_size):
	data,label=fetch_train_data(target_train_index,i,1)
	data=Variable(data)
	result=net(data.cuda())
	_,predicted=torch.max(result,1)
	label=label[0]
	_,result=torch.max(Attack_models[label](result[0].unsqueeze(0).cuda()),1)
	if result.data[0]==1:
		correct[label]+=1
		total[label]+=1
for i in range(0,set_size):
	data,label=fetch_test_data(target_test_index,i,1)
	data=Variable(data)
	result=net(data.cuda())
	_,predicted=torch.max(result,1)
	label=label[0]
	_,result=torch.max(Attack_models[label](result[0].unsqueeze(0).cuda()),1)
	if result.data[0]==1:
		total[label]+=1
print correct
print total
for i in range(10):
	print correct[i]/total[i]