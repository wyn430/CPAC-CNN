"""
This is the GPU implementation of regular CNN model using pytorch on MNIST. It is the baseline model used to compare with CPAC-CNN. The codes are similar to train_mnist.py except for the model initilization.
python train_mnist_cnn.py [number of filters] [filter_size] [rank] [epochs] [device]; python train_mnist_cnn.py 8 3 6 10 cuda:0
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
import numpy as np
np.random.seed(48)
torch.manual_seed(48)
from conv_decomp_torch import Conv_Decomp
from cnn_torch import CNN_Decomp
from cnn_torch import CNN

import time
import sys
import os
import warnings
warnings.filterwarnings("ignore")

##MNIST
##get parameters from command line
num_filters = int(sys.argv[1])
filter_h = int(sys.argv[2])
filter_w = int(sys.argv[2])
rank = int(sys.argv[3]) ##this parameter is useless in regular cnn
n_epochs = int(sys.argv[4])
cuda_device = str(sys.argv[5])

image_channels = 1
num_class = 10
devices = [torch.device(cuda_device),torch.device("cpu")]
learning_rate = 0.0001
batch_size_train = 1
batch_size_test = 1
log_interval = 100


##data loader
def load_dataset():
    data_path = 'data/'
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    
    return train_loader, test_loader

##model training
def train(epoch,file_root,file_name):
    Net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = Net(data[:,0:1,:,:].to(devices[0]))
        loss = F.nll_loss(output.to(devices[1]), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item())
            print(log)
            train_losses.append(loss.item())
            train_counter.append(
               (batch_idx*1) + ((epoch-1)*len(train_loader.dataset)))
            
##model testing            
def test(test_acc,file_root,file_name):
    Net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
          output = Net(data[:,0:1,:,:].to(devices[0]))
          test_loss += F.nll_loss(output.to(devices[1]), target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1].to(devices[1])
          correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    log = '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset))
    with open(file_root+'/'+file_name, 'a') as f:
          f.write(log+'\n')
    print(log)
    ##save the model with best testing accuracy
    if 100. * correct / len(test_loader.dataset) > test_acc:
        test_acc = 100. * correct / len(test_loader.dataset)
        torch.save(Net.state_dict(), file_root + '/' + 'model.pth')
        torch.save(optimizer.state_dict(), file_root + '/' + 'optimizer.pth')
    return test_acc
    
##result directory    
par_root = "./result"
file_root = par_root + '/' + 'mnist_2layer_cnn_'+str(num_filters) + '_' + str(rank) + '_' + str(n_epochs) + '_' + time.strftime("%Y%m%d-%H%M%S")
os.mkdir(file_root)  
file_name = 'mnist_2layer_'+str(num_filters) + '_' + str(rank) + '_' + str(n_epochs) + '.txt'


##initialize data loader
train_loader, test_loader = load_dataset()

##get image size
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
input_shape = example_data.shape

##initialize model CNN, the detailed model structure can be modified in cnn_torch.py, currently there are two conv layers
#Net = CNN_Decomp(num_filters, filter_h, filter_w, image_channels, rank, devices, num_class, input_shape) ##CPAC-CNN
Net = CNN(num_filters, filter_h, filter_w, image_channels, rank, devices, num_class, input_shape)
Net.to(devices[0])
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
    
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

test_acc = 0
##train and test
for epoch in range(1, n_epochs + 1):
  train(epoch,file_root,file_name)
  test_acc = test(test_acc,file_root,file_name)
