"""
This code includes the CPU implementation of CPAC-Conv layer 
"""


import numpy as np
np.random.seed(48)

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from tensorly.decomposition import parafac
from tensorly.tenalg import mode_dot
from tensorly import unfold
import tensorly as tl
import pickle

import time


class Conv3x3:

  def __init__(self, num_filters, filter_h, filter_w, image_channels, rank):
    
    self.image_channels = image_channels
    self.num_filters = num_filters
    self.filter_h = filter_h
    self.filter_w = filter_w
    self.rank = rank
    
    self.filters = np.random.randn(num_filters, filter_h, filter_w, image_channels) / (filter_h*filter_w)
    
    tensor = tl.tensor(self.filters)
    ##initialize the cp-decomposed convolutional factors
    self.factors = parafac(tensor, rank)
    self.filters_recon = tl.kruskal_to_tensor((self.factors))
    
    ##initialize moments and parameters for adam
    self.v0 = np.zeros(self.factors[0].shape)
    self.v1 = np.zeros(self.factors[1].shape)
    self.v2 = np.zeros(self.factors[2].shape)
    self.v3 = np.zeros(self.factors[3].shape)
    self.v = [self.v0,self.v1,self.v2,self.v3]
    
    self.s0 = np.zeros(self.factors[0].shape)
    self.s1 = np.zeros(self.factors[1].shape)
    self.s2 = np.zeros(self.factors[2].shape)
    self.s3 = np.zeros(self.factors[3].shape)
    self.s = [self.s0,self.s1,self.s2,self.s3]
    
    self.beta1 = 0.99
    self.beta2 = 0.999
  

  def forward(self, input):
    '''
    perform the forward propagation of CPAC-Conv layer
    input: images with shape (h,w,channels) or (h,w)
    '''
    
    self.last_input = input
    
    if self.image_channels == 1:
        h,w = input.shape
        reshape_input = image.extract_patches_2d(input, (self.filter_h, self.filter_w))
        p,_,_ = reshape_input.shape
        reshape_input = reshape_input.reshape(p,self.filter_h, self.filter_w,1)
    else:
        h,w,_ = input.shape
        reshape_input = image.extract_patches_2d(input, (self.filter_h, self.filter_w))
        p,_,_,_ = reshape_input.shape
        
    
    for i in range(self.rank):
    
        result = mode_dot(reshape_input,self.factors[3][:,i],mode=3)
        result = mode_dot(result,self.factors[2][:,i],mode=2)
        result = mode_dot(result,self.factors[1][:,i],mode=1)
        result = np.outer(result,self.factors[0][:,i])
    
        if i == 0:
            output = result
        else:
            output += result
            
    output = output.reshape((h-2,w-2,self.num_filters))
        

    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    perform the backward propagation of CPAC-Conv layer
    '''
    
    if len(self.last_input.shape)==3:
        input_w, input_h, input_c = self.last_input.shape
    else:
        input_w, input_h = self.last_input.shape
        input_c = 1
        
    h,w,c = d_L_d_out.shape    
    d_L_d_currout = d_L_d_out
    d_L_d_preout = np.zeros((input_w, input_h, input_c))
    self.filters = tl.kruskal_to_tensor((self.factors))
    
    for curr_f in range(c):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + self.filter_h <= input_h:
            curr_x = out_x = 0
            while curr_x + self.filter_h <= input_h:
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                d_L_d_preout[curr_y:curr_y+self.filter_h, curr_x:curr_x+self.filter_h, :] += d_L_d_currout[out_y,out_x,curr_f] * self.filters[curr_f] 
                curr_x += 1
                out_x += 1
            curr_y += 1
            out_y += 1
            
            
    d_L_d_factor0 = np.zeros(self.factors[0].shape) #(8,5)
    d_L_d_factor1 = np.zeros(self.factors[1].shape) #(3,5)
    d_L_d_factor2 = np.zeros(self.factors[2].shape) #(3,5)
    d_L_d_factor3 = np.zeros(self.factors[3].shape) #(1,5)
     
        
    if self.image_channels == 1:
        last_input_reshape = image.extract_patches_2d(self.last_input, (self.filter_h, self.filter_w))
        p,_,_ = last_input_reshape.shape
        n = 1
        last_input_reshape = last_input_reshape.reshape((p,self.filter_h, self.filter_w, n))
        
    else:
        last_input_reshape = image.extract_patches_2d(self.last_input, (self.filter_h, self.filter_w))
        p,_,_,n = last_input_reshape.shape
        
    d_L_d_out = d_L_d_out.reshape((p,self.num_filters))
    
    for i in range(self.rank):
        
        ##update K_r^N, d_L_d_factor0
        A_3 = mode_dot(last_input_reshape,self.factors[3][:,i],mode=3)
        A_2 = mode_dot(A_3,self.factors[2][:,i],mode=2)
        A_1 = mode_dot(A_2,self.factors[1][:,i],mode=1)
        A_1 = A_1.reshape((p,1))
        I_1 = np.identity(self.num_filters)
        d_L_d_factor0[:,i] = np.sum((d_L_d_out * A_1), axis=0)

        
        ##update K_r^X, d_L_d_factor1
        d_L_d_factor1[:,i] = np.sum(np.kron(self.factors[0][:,i:i+1],
                                            A_2) * d_L_d_out.reshape((p*self.num_filters,1)), 
                                    axis=0)
        
        
        ##update K_r^Y, d_L_d_factor2

        A_3_unfold = unfold(A_3,2)
        I_2 = np.identity(p)
        B_2 = np.outer(self.factors[1][:,i],self.factors[0][:,i])
        B2I2 = np.kron(B_2,I_2)
        d_L_d_factor2[:,i] = np.sum(np.dot(A_3_unfold, B2I2) * d_L_d_out.reshape((p*self.num_filters,1)).T, 
                                    axis=1)
        
        
        ##update K_r^S, d_L_d_factor3 
        U_unfold = unfold(last_input_reshape,3)        
        I_3 = np.identity(self.filter_h*p)
        
        d_L_d_factor3[:,i] = np.sum(np.dot(U_unfold, 
                                           np.dot(np.kron(self.factors[2][:,i:i+1],I_3), 
                                                  B2I2)) * d_L_d_out.reshape((p*self.num_filters,1)).T,
                                    axis=1)
    
    ##adam
    d_factors = [d_L_d_factor0,d_L_d_factor1,d_L_d_factor2,d_L_d_factor3]
    # Update filters
    for i in range(len(self.factors)):
        self.v[i] = self.beta1*self.v[i] + (1-self.beta1)*d_factors[i]
        self.s[i] = self.beta2*self.s[i] + (1-self.beta2)*d_factors[i]**2
        self.factors[i] -= learn_rate * self.v[i]/np.sqrt(self.s[i]+1e-7)
        
 

    
    return d_L_d_preout

  def save_weights(self, root):
        
    """
    save the current weights
    """
    save_path_factor0 = root + '/' + "factor0.pkl"
    save_path_factor1 = root + '/' + "factor1.pkl"
    save_path_factor2 = root + '/' + "factor2.pkl"
    save_path_factor3 = root + '/' + "factor3.pkl"
    
    
    with open(save_path_factor0, 'wb') as file:
      pickle.dump(self.factors[0], file)
    with open(save_path_factor1, 'wb') as file:
      pickle.dump(self.factors[1], file)
    with open(save_path_factor2, 'wb') as file:
      pickle.dump(self.factors[2], file)
    with open(save_path_factor3, 'wb') as file:
      pickle.dump(self.factors[3], file)
        
        
    
