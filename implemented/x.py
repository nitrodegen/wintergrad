import torch.nn as nn
import torch
import numpy as np
import os,io,sys
import torchvision


a=[0,1,2,3,4,5,6,7]
b=[0,1,2,3,4,5,6,7]

def lgls(pyi,yi):
    N = pyi.size
    yi = yi[0]
    pyi=pyi[0]
    print(yi,pyi)
    calc =   -(1/N)*(yi*np.log(pyi)+(1-yi)*np.log(1-pyi))
    return calc 

class BDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Sequential(
            nn.Linear(1,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        x = self.b(x)
        print(x.shape)
        return x


def crossloss(y,yi):
    res = [] 
    bbk = 0 
    for i in range(len(yi)):
    
     
        ba = -( y[0]*np.log(yi[i]) +(1-y[0])*np.log(1-yi[i]))
        bbk+=ba

        
    print(bbk)
   # exit(1)


xd = np.array([0.115013,0.110342,0.107984,0.114351,0.110572,0.111032,0.110059,0.108831,0.111816])
haha = np.array([2])

crossloss(haha,xd)

dodik = nn.CrossEntropyLoss()
test = torch.tensor([0.115013,0.110342,0.107984,0.114351,0.110572,0.111032,0.110059,0.108831,0.111816]).float()
bb = torch.tensor(2)
print(dodik(test,bb))
