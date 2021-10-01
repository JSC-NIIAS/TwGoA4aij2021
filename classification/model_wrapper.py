import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class Model_head(nn.Module):
    def __init__(self,num_classes,in_features):
        super(Model_head, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(num_classes)):
            self.layers.append(nn.Linear(in_features,num_classes[i],bias=True))

    def forward(self,x):
        return [self.layers[i](x) for i in range(len(self.layers))]

class Classification_model(nn.Module):
    def __init__(self,base_model,model_type,num_classes):
        super(Classification_model,self).__init__()
        self.model=base_model
        self.in_features=self.model.fc.in_features
        self.model_type = model_type
        self.model.fc = nn.Linear(self.in_features,num_classes,bias=True)

    def forward(self,x):
        out=self.model(x)
        return out
