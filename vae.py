import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn2
from torchvision.utils import save_image
import torchvision.transforms as transforms
import math

mb_size = 100
Z_dim = 100
X_dim = 32*32
y_dim = 10
h_dim = 128
c =0
C=1000.
lr = 1e-3

transform = transforms.Compose([transforms.Scale(32),
                                transforms.ToTensor()])
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=True, download=True,transform=transform),batch_size=mb_size,shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=False, transform=transform),batch_size=mb_size,shuffle=True)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X):
    h = nn.relu(torch.mm(X, Wxh) + bxh.repeat(X.size(0), 1))
    z_mu = torch.mm(h,Whz_mu) + bhz_mu.repeat(h.size(0), 1)
    z_var = torch.mm(h,Whz_var) + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var
def sample_z(mu, log_var):
    # Using reparameterization trick to sample from a gaussian
    eps = Variable(torch.randn(mu.size(0), Z_dim))
    tmp=mu + torch.exp(log_var / 2) * eps
    norm=float(np.linalg.norm(tmp.data.numpy()))
    val=0.0
    if norm>C:
        val=C*tmp/norm
    else:
        val=tmp
    return val
def encoder(X):
    #h = nn.relu(torch.mm(X, Wxh) + bxh.repeat(X.size(0), 1))
    #z_mu = torch.mm(h,Whz_mu) + bhz_mu.repeat(h.size(0), 1)
    mu,var=Q(X)

    return sample_z(mu,var)

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z):
    h = nn.relu(torch.mm(z, Wzh) + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(torch.mm(h, Whx) + bhx.repeat(h.size(0), 1))
    return X
params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]
solver = optim.Adam(params, lr=lr)

for it in range(5):
    for i, data in enumerate(trainloader,0):
        solver.zero_grad()
        X,_=data
        X=X.view(-1,32*32)
        X = Variable(X)
        #forward
        z_mu, z_var = Q(X)
        z = sample_z(z_mu, z_var)
        X_sample = P(z)
        #loss
        recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
        #kl_loss = 0.5 * torch.sum(torch.exp(z_var) + - 1. - z_var)
        loss = recon_loss + kl_loss
        #back
        loss.backward()
        if i%100==0:
            print ("Epoch:{}, Iter:{}, Loss val:{}".format(it,i,loss.data[0]))
        # Update
        solver.step()
        # Print and plot every now and then
        if i % 1000 == 0:
            z = Variable(torch.randn(mb_size, Z_dim))
            samples = P(z).data.numpy()[:16]
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(32, 32), cmap='Greys_r')
            if not os.path.exists('./out/'):
                os.makedirs('./out/')
            plt.savefig('./out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
            c += 1
            plt.close(fig)
#test
class Net(nn2.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn2.Linear(100,10)
    def forward(self,x):
        x=F.sigmoid(self.fc1(x))
        return x
server=Net()
criterion=nn2.CrossEntropyLoss()
optimizer_server=optim.Adam(server.parameters(), lr=lr, betas=(0.5, 0.999))


new_training_data=torch.FloatTensor([])
new_training_label=torch.LongTensor([])
for i, data in enumerate(trainloader,0):
    x,labels=data
    new_training_data=torch.cat((new_training_data,x),0)
    new_training_label=torch.cat((new_training_label,labels),0)

##adding noise
epsilon=1.0
delta=1e-5
theta=((2*math.log(1.25/delta))**0.5)/epsilon
new_training_data=new_training_data.view(-1,32*32)
new_training_data=Variable(new_training_data)
new_training_data=encoder(new_training_data).data
new_training_data+=torch.normal(torch.zeros(new_training_data.size()),std=torch.ones(new_training_data.size())*C*C*theta*theta)

for it in range(10):
    for i in range(60000/100):
        X,labels=new_training_data[i*100:(i+1)*100],new_training_label[i*100:(i+1)*100]
        X,labels= Variable(X),Variable(labels)
        output=server(X)
        optimizer_server.zero_grad()
        loss=criterion(output,labels)
        loss.backward()
        optimizer_server.step()
        if i %100==0:
            print("epoch:{}, iter:{}, loss:{}".format(it+1,i+1,loss.data[0]/50.0))

def test():
    correct=0.
    total=0.
    for i , data in enumerate(testloader,0):
        images,labels=data
        images=Variable(images).view(-1,32*32)
        newinput=encoder(images)
        outputs=server(newinput)
        _,prediction=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(prediction==labels).sum()
    print('Accuracy of the network on the 10000 test images:{}'.format(correct / total))
test()
