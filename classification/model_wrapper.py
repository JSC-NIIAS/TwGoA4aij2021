import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

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
