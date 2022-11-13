import torch.nn as nn
import torch
import numpy as np
import os,io,sys
import torchvision
import torchvision.transforms as transforms


data = torchvision.datasets.MNIST("./data/",train=True,download=True,transform=transforms.Compose([
                               transforms.Resize(28),                  
                               transforms.ToTensor(),
                           ]))
ind= torch.arange(2048)
data = torch.utils.data.Subset(data,ind)
trainl = torch.utils.data.DataLoader(data,batch_size=256,shuffle=True)
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10),
          #  nn.Softmax(),
        )
        
    def forward(self,x):
        x = self.lin(x)
        print(x)
        
        return x


"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin =nn.Linear(784,256).weight
    def forward(self,x):
        x =self.lin
        print(x)
        exit(1)

        return x

d = Net()
optim = torch.optim.Adam(d.parameters(),lr=3e-4)
for e in range(20):
    dg=0
    for b,(x,y) in enumerate(trainl):
            y =y[b]
            x = x[b].view(-1)
            g = d(x).reshape(10)
            ls = nn.functional.cross_entropy(g,y)
            ls.backward()
            optim.step()
            dg+=1
inp = 0 
tete = 0 
for b,(x,y) in enumerate(trainl):
    inp = x
    tete = y
    break

inp =inp[0].view(-1)
g = np.argmax(d(inp).cpu().detach().numpy(),axis=-1)
print(g)
print(tete[0])


