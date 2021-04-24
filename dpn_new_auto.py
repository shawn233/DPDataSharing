'''Dual Path Networks in PyTorch.'''
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from new_auto import autoencoder
from new_auto import encode
from torchvision.utils import save_image
import torchvision.transforms as transforms

##load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=32,shuffle=False, num_workers=2)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


##functions to show an image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x
##function to calculate accuracy 
def accuracy(label,output):
    _,prediction=torch.max(output.data,1)
    return (prediction==label).sum()

##model definition
autoencoder=autoencoder().cuda()
criterion_auto=nn.MSELoss().cuda()
optimizer_auto=optim.Adam(autoencoder.parameters(),lr=0.01)
#dpn92
class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DPN26():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)
dpn=DPN92().cuda()
criterion=nn.CrossEntropyLoss().cuda()
optimizer=optim.Adam(dpn.parameters(),lr=0.01,weight_decay=1e-5)

##training auto-encoder
for epoch in range(0):
    for i, data in enumerate(trainloader,0):
        img,_=data
        newimg=img[:,0].numpy()
        newimg=torch.Tensor(np.reshape(newimg,(-1,32*32)))
        newimg = Variable(newimg).cuda()

        optimizer_auto.zero_grad()
        output = autoencoder(newimg)
        loss = criterion_auto(output, newimg)

        loss.backward()
        optimizer_auto.step()
        if i %100==0:
            print("epoch:{}, iter:{}, loss:{}".format(epoch,i,loss.data[0]))
print("finished autoencoder training")


##training dpn
for epoch in range(1):
    for i,data in enumerate(trainloader,0):
        image,label=data
        image=image.cuda()
        encoded_data=encode(autoencoder,image)
        encoded_data=encoded_data.view(-1,3,30,30)
        image,label=Variable(encoded_data),Variable(label)
        output=dpn(image)
        loss=criterion(output,label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i%100==0):
            print("Epoch:{}, Round:{},  loss:{}".format(epoch,i,loss.data[0]))
    testdata=iter(testloader)
    image,label=testdata.next()
    encoded_data=encode(autoencoder,image).view(-1,3,30,30)
    image=Variable(encoded_data)
    output=dpn(image.cuda())
    num=accuracy(label.cuda(),output)
    print("Epoch:{} ends,  Accuracy:{}%".format(epoch,100*num/float(label.size(0))))
