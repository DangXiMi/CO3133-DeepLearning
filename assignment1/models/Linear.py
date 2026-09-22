import sys

import torch
import torch.nn as nn
import numpy as np

class LinearClassifer(nn.Module):
    def __init__(self, num_class = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.f = nn.Linear(28*28,num_class)
        
    def forward(self,x):
        return self.f(self.flatten(x))
    
    def num_para(self): 
        return sum(p.numel() for p in self.f.parameters() if p.requires_grad)