import torch
from torch import nn
from torch.autograd import Variable as V
import numpy as np
np.random.seed(48)
torch.manual_seed(48)
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from tensorly.decomposition import parafac
from tensorly.tenalg import mode_dot
from tensorly import unfold
import tensorly as tl
import pickle

import time

class Conv_Decomp(nn.Module):
    def __init__(self, num_filters, filter_h, filter_w, image_channels, rank, devices):
        # nn.Module.__init__(self)
        super(Conv_Decomp, self).__init__()
        ##initialize factors
        filters = np.random.randn(num_filters, filter_h, filter_w, image_channels) / (filter_h*filter_w)
        tensor = tl.tensor(filters)
        factors = parafac(tensor, rank)
        
        self.factor0 = nn.Parameter(torch.tensor(factors[0]).to(devices[0]))
        self.factor1 = nn.Parameter(torch.tensor(factors[1]).to(devices[0]))
        self.factor2 = nn.Parameter(torch.tensor(factors[2]).to(devices[0]))
        self.factor3 = nn.Parameter(torch.tensor(factors[3]).to(devices[0]))
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.num_filters = num_filters
        self.image_channels = image_channels
        self.rank = rank
        self.devices = devices      
         
    def forward(self, input):
        
    
        
            
        input_shape = input.shape #C,H,W
        reshape_input = input.unfold(1, self.filter_h, 1).unfold(2, self.filter_w, 1)
        shape = reshape_input.shape
        reshape_input = reshape_input.reshape(shape[0],shape[1]*shape[2],shape[3],shape[4])#C,N,h,w
        reshape_input = reshape_input.permute(1,2,3,0).to(self.devices[0])#N,h,w,C


        for i in range(self.rank):
            result = torch.einsum('abcd,d->abc', reshape_input.float(), self.factor3[:,i].float())
            result = torch.einsum('abc,c->ab', result, self.factor2[:,i].float())
            result = torch.einsum('ab,b->a', result, self.factor1[:,i].float())
            result = torch.einsum('a,b->ab', result, self.factor0[:,i].float())
            #result = mode_dot(reshape_input,self.factors[3][:,i],mode=3)
            #result = mode_dot(result,self.factors[2][:,i],mode=2)
            #result = mode_dot(result,self.factors[1][:,i],mode=1)
            #result = np.outer(result,self.factors[0][:,i])

            if i == 0:
                output = result
            else:
                output += result

        output = output.reshape((input_shape[1]-2,input_shape[2]-2,self.num_filters))
        output = output.permute(2,0,1)

        return output
        
