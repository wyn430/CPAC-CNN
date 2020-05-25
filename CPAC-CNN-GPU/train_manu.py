"""
This is the GPU implementation of regular CPAC-CNN model using pytorch on Manufacturing.
python train_manu.py [number of filters] [filter_size] [rank] [epochs] [device]; python train_manu.py 8 3 6 10 cuda:0
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

##Manu
##get parameters from command line
num_filters = int(sys.argv[1])
filter_h = int(sys.argv[2])
filter_w = int(sys.argv[2])
rank = int(sys.argv[3])
n_epochs = int(sys.argv[4])
cuda_device = str(sys.argv[5])

image_channels = 1
num_class = 6
devices = [torch.device(cuda_device),torch.device("cpu")]
learning_rate = 0.0001
batch_size_train = 1
batch_size_test = 1
log_interval = 100

##data loader
def load_dataset():
    data_path = '../Data/Magnetic-tile-defect-datasets_torch/'
    
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5,), (0.5,))])
    )
    
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    
    return train_loader, test_loader

##model training
def train(epoch,file_root,file_name):
    Net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output = Net(data[0,0:1,:,:]) ##the reshape process need to be done on CPU
        loss = F.nll_loss(output.to(devices[1]), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item())
            #with open(file_root+'/'+file_name, 'a') as f:
                #f.write(log+'\n')
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
          output = Net(data[0,0:1,:,:])
          #output = Net(data[:,0:1,:,:].to(devices[0]))
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
    if test_loss > test_acc:
    #if 100. * correct / len(test_loader.dataset) > test_acc:
        test_acc = 100. * correct / len(test_loader.dataset)
        torch.save(Net.state_dict(), file_root + '/' + 'model.pth')
        torch.save(optimizer.state_dict(), file_root + '/' + 'optimizer.pth')
    return test_acc
    
    
##result directory   
par_root = "./result"
file_root = par_root + '/' + 'manu_2layer_'+str(num_filters) + '_' + str(rank) + '_' + str(n_epochs) + '_' + time.strftime("%Y%m%d-%H%M%S")
os.mkdir(file_root)  
file_name = 'manu_2layer_'+str(num_filters) + '_' + str(rank) + '_' + str(n_epochs) + '.txt'

##initialize data loader    
train_loader, test_loader = load_dataset()

##get image size
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
input_shape = example_data.shape

##initialize model CPAC-CNN, the detailed model structure can be modified in cnn_torch.py, currently there are two conv layers
Net = CNN_Decomp(num_filters, filter_h, filter_w, image_channels, rank, devices, num_class, input_shape)
#Net = CNN(num_filters, filter_h, filter_w, image_channels, rank, devices, num_class, input_shape)
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
