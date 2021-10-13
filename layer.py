
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd.function import Function
import numpy as np
import math




class ANEPN(nn.Module):
    def __init__(self, M, H, C):
        super(ANEPN,self).__init__()
        self.Dense1 = nn.Linear(M, H)
        self.Dense2 = nn.Linear(H, C)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self,x):      
        x = self.Dense1(x)    
        x_norm = t.norm(x,dim=0,keepdim=True) + 1e-12
        z =  x/x_norm
        x = self.Dense2(x)
        return self.logsoftmax(x), z


