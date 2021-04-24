import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from auto_cifar import autoencoder
from torchvision.utils import save_image
import torchvision.transforms as transforms

##load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

##functions to show an image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


##model definition
#autoencoder
autoencoder=autoencoder()
optimizer_auto=optim.Adam(autoencoder.parameters(),lr=0.001,weight_decay=1e-5)
def criterion_auto(input,target):
	return torch.mean((input-target)**2)
#neural network(CNN)
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Conv2d(32,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		self.fc1=nn.Linear(16*4*4,120)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)
	def forward(self,x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.pool(F.relu(self.conv2(x)))
		x=x.view(-1,16*4*4)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x 
net=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

##training auto-encoder
for epoch in range(10):
	for i, data in enumerate(trainloader,0):
		inputs,labels=data
		inputs,labels=Variable(inputs),Variable(labels)
		optimizer_auto.zero_grad()
		output=autoencoder(inputs)
		loss=criterion_auto(output,inputs)
		loss.backward()
		optimizer_auto.step()
		if i %100==0:
			print("epoch:{}, iter:{}, loss:{}".format(epoch,i,loss.data[0]))
			pic = to_img(output.data)
			save_image(pic, './dc_img/image_{}.png'.format(epoch))
print("finished autoencoder training")
for i in range()

'''

##training cnn
for epoch in range(50):
	for i, data in enumerate(trainloader,0):
		inputs,labels=data
		inputs,labels=Variable(inputs),Variable(labels)
		newinputs=autoencoder.encoder(inputs)
		print newinputs.size()
		optimizer.zero_grad()
		output=net(newinputs)
		loss=criterion(output,labels)
		loss.backward()
		optimizer.step()
		if i% 100==0:
			print("epoch:{}, iter:{}, loss:{}".format(epoch,i,loss.data[0]))
print("finished classifier training")

correct=0
total=0

for data in testloader:
	images,labels=data
	newinput=autoencoder.encoder(Variable(images))
	outputs=net(newinput)
	_,prediction=torch.max(outputs.data,1)
	total+=labels.size(0)
	correct+=(prediction==labels).sum()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


class_correct=list(0. for i in range(10))
class_total=list(0. for i in range(10))
for data in testloader:
	images,labels=data
	newinput=autoencoder.encoder(Variable(images))
	outputs=net(newinput)
	_,predicted=torch.max(outputs.data,1)
	c=(predicted==labels).squeeze()
	for i in range(4):
		label=labels[i]
		class_correct[label]+=c[i]
		class_total[label]+=1
for i in range(10):
	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

'''

