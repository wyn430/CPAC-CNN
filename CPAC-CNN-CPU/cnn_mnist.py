"""
Training and save the CPAC-CNN model on mnist dataset, limited by computational cost, we only test 1-layer CPAC-CNN with smaller dataset, it can be easily extended to multiple-layer CPAC-CNN (example extension is commented in code)

python cnn_mnist.py [number of filter] [rank] [number of epoch]
"""


import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import time
import sys
from tools import *
np.random.seed(48)


##get parameters from command line
num_filters = int(sys.argv[1])
rank = int(sys.argv[2])
num_epoch = int(sys.argv[3])

filter_h = 3
filter_w = 3
N = 100 ##define the log print interval

print("preparing data")

train_images = mnist.train_images()[:2000]
train_labels = mnist.train_labels()[:2000]
test_images = mnist.test_images()[:300]
test_labels = mnist.test_labels()[:300]

permutation = np.random.permutation(len(train_images))

train_images = train_images[permutation]
train_labels = train_labels[permutation]
num_class = np.max(train_labels) + 1


if len(train_images[0].shape) == 2:
    h,w = train_images[0].shape
    image_channels = 1
else:
    h,w,image_channels = train_images[0].shape

##model initialization
conv = Conv3x3(num_filters,filter_h,filter_w,image_channels,rank)
pool = MaxPool2()
softmax = Softmax(int((h-2)/2) * int((w-2)/2) * num_filters, num_class)
#####uncommented this block if want to check 2-layer CPAC-CNN######
#conv2 = Conv3x3(num_filters,filter_h,filter_w,num_filters,rank) # 28x28x1 -> 26x26x8
#softmax = Softmax(int(((h-2)/2 - 2)) * int(((w-2)/2 - 2)) * num_filters, num_class)
###################################################################


##result directory
par_root = "./result"
file_root = par_root + '/' + 'mnist_single_'+str(num_filters) + '_' + str(rank) + '_' + str(num_epoch) + '_' + time.strftime("%Y%m%d-%H%M%S")
os.mkdir(file_root) 

##log file
file_name = 'mnist_single_'+str(num_filters) + '_' + str(rank) + '_' + str(num_epoch) + '.txt'

def forward(image, label):
  '''
  forward propagation of CPAC-CNN
  '''
  ##standardize input
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
#  out = conv2.forward(out)  ##for 2-layer CPAC-CNN
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, num_class, lr=.005):
  '''
  training process (forward and backward)
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(num_class)
  gradient[label] = -1 / out[label]

  # Backward
  gradient = softmax.backprop(gradient, lr)
#  gradient = conv2.backprop(gradient, lr) ##for 2-layer CPAC-CNN
  gradient = pool.backprop(gradient) 
  gradient = conv.backprop(gradient, lr)
  

  return loss, acc

print('CPAC-CNN initialized!')
max_acc = 0
# Train the CNN for 3 epochs
start = time.time()
for epoch in range(num_epoch):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train
  loss = 0
  num_correct = 0
  num_samples = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    
    if i % N == N-1:
      log = '[Step %d] Past %d steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, N, loss / N, num_correct)
    
      with open(file_root+'/'+file_name, 'a') as f:
        f.write(log+'\n')

      print(log)
      loss = 0
      num_correct = 0
    
    l, acc = train(im, label, num_class)
    
    loss += l
    num_correct += acc
    num_samples += 1
    
  print('\n--- Testing the CNN ---')
  loss = 0
  num_correct = 0
  for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

  num_tests = len(test_images)
  test_loss = 'Test Loss:' + str(loss / num_tests)
  test_acc = 'Test Accuracy:' + str(num_correct / num_tests)
  ##if the model accuracy improved, save the current weights
  if num_correct / num_tests > max_acc:
        max_acc = num_correct / num_tests
        softmax.save_weights(file_root)
        conv.save_weights(file_root)
        
  with open(file_root+'/'+file_name, 'a') as f:
        f.write(test_loss+'\n')
        f.write(test_acc+'\n')
  print(test_loss)
  print(test_acc)

# final test of CPAC-CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
end = time.time()
test_loss = 'Test Loss:' + str(loss / num_tests)
test_acc = 'Test Accuracy:' + str(num_correct / num_tests)
time = 'Time Elapse:' + str(end-start)

print(test_loss)
print(test_acc)
print(time)

with open(file_root+'/'+file_name, 'a') as f:
        f.write(test_loss+'\n')
        f.write(test_acc+'\n')
        f.write(time+'\n')
