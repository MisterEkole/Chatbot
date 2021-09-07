import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, inputs, hidden, num_classes):
        super(Network, self).__init__()
        self.l1= nn.Linear(inputs, hidden)
        self.l2= nn.Linear(hidden, hidden)
        self.l3=nn.Linear(hidden, num_classes)
      
        self.relu=nn.ReLU()
        
    def forward(self, x):
        out= self.l1(x)
        out=self.relu(out)
        out= self.l2(out)
        out=self.relu(out)
        out= self.l3(out)
      
        return out
        