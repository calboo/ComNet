
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Conv1d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, groups=1, bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))
    
    
class TemporalBlock(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, activation, kernel_size, stride, dilation, dropout=0.2):
       
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(CausalConv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation))
        
        if activation == "gelu":
            self.activ1 = nn.GELU()
        elif activation == "relu":
            self.activ1 = nn.ReLU()
        else:
            self.activ1 = nn.LeakyReLU()
            
        self.dropout1 = nn.Dropout1d(dropout)

        self.conv2 = weight_norm(CausalConv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation))

        if activation == "gelu":
            self.activ2 = nn.GELU()
        elif activation == "relu":
            self.activ2 = nn.ReLU()
        else:
            self.activ2 = nn.LeakyReLU()
            
        self.dropout2 = nn.Dropout1d(dropout)

        self.net = nn.Sequential(self.conv1, self.activ1, self.dropout1,
                                 self.conv2, self.activ2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        
        self.out_shape= n_outputs
        
        self.init_weights()

    def init_weights(self):
        
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)
    
    
class TemporalConvNet(nn.Module):
    
    def __init__(self, num_inputs, num_channels, activation, kernel_size=3, dropout=0.2):
        
        super(TemporalConvNet, self).__init__()
        
        self.layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers += [TemporalBlock(in_channels, out_channels, activation, kernel_size,
                                     stride=1, dilation=dilation_size, dropout=dropout)]

        self.out_shape = self.layers[-1].out_shape
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        
        return self.network(x)
    
    
class TCN(nn.Module):
    
    def __init__(self, history_length, num_inputs, num_channels, activation, kernel_size=3, dropout=0.2):
        
        super(TCN, self).__init__()
        
        self.tcn = TemporalConvNet(num_inputs, num_channels, activation, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(self.tcn.out_shape*history_length, 2)
        
        self.init_weights()
        self.device = next(self.parameters()).device

    def init_weights(self):

        self.linear.weight.data.normal_(0, 0.01)
    
    def set_device(self,device):
        
        self.to(device)
        self.device = device
    
        return self

    def count_parameters(self):
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return '{:.4f} million parameters'.format(n_params/1e6)

    def forward(self, x):
        
        y = x.transpose(-2,-1)
        y = self.tcn(y)
        y = self.linear(y.flatten(-2,-1))
        
        means = y[:,0]
        variances = (y[:,1]+torch.exp(-y[:,1]/2))*torch.sigmoid(y[:,1])

        return means, variances
    
    