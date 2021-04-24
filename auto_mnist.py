import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

##load data
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=True, download=True,transform=transforms.ToTensor()),
    batch_size=50, shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=False, transform=transforms.ToTensor()),
    batch_size=50, shuffle=True)

##functions to show an image
def plot(samples,name,size):
	plt.figure(figsize=(size,size))
	fig=plt.imshow(samples,origin='upper',cmap='gray')
	plt.savefig(name, bbox_inches='tight')
	plt.close()
##model definition
#autoencoder
class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder,self).__init__()
		self.fc1=nn.Linear(28*28,400)
		self.fc2=nn.Linear(400,256)
		self.fc3=nn.Linear(256,400)
		self.fc4=nn.Linear(400,28*28)
	def forward(self,x):
		x=F.sigmoid(self.fc1(x))
		x=F.sigmoid(self.fc2(x))
		x=F.sigmoid(self.fc3(x))
		x=F.sigmoid(self.fc4(x))
		return x
autoencoder=Autoencoder()
criterion_auto=nn.MSELoss()
optimizer_auto=optim.Adam(autoencoder.parameters(),lr=0.01)

##training auto-encoder
for epoch in range(10):
	running_loss=0.0
	for i, data in enumerate(trainloader,0):
		inputs,labels=data
		inputs,labels=Variable(inputs),Variable(labels)
		optimizer_auto.zero_grad()
		inputs=inputs.view(-1,28*28)
		output=autoencoder(inputs)
		loss=criterion_auto(output,inputs)
		loss.backward()
		optimizer_auto.step()
		running_loss+=loss.data[0]
		if i %100==0:
			print("epoch:{}, iter:{}, loss:{}".format(epoch+1,i+1,running_loss/2000.0))
			running_loss=0. 
			data=iter(testloader)
			images,labels=data.next()
			images=autoencoder(Variable(images.view(-1,784)))
			images=images.data.view(-1,28,28)
			images=images[0:16].numpy()
			drawing=np.zeros((4*28,4*28))
			for k in range(4):
				for j in range(4):
					drawing[k*28:(k+1)*28,j*28:(j+1)*28]=images[k*4+j]
			plot(drawing,'image_'+str(i)+'.png',2)
print("finished autoencoder training")