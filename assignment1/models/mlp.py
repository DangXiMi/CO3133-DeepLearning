import sys

import torch
import torch.nn as nn
import numpy as np

# Models

class MLP(nn.Module):
    def __init__(self,num_class = 10, hidden_layer=(256,128), ):
        super().__init__()
        self.flatten = nn.Flatten()
        
        
        layers, feature = [], 28*28
        for hidden in hidden_layer:
            layers += [nn.Linear(feature,hidden),nn.ReLU()]
            feature = hidden
        layers += [nn.Linear(feature,num_class)]
        
        self.network = nn.Sequential(*layers)
            
        
    def forward(self,x):
        return self.network(self.flatten(x))
    
    def num_para(self): 
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    