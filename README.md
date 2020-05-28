# CPAC-CNN

Implementation of our recent paper, CPAC-CNN: CP-decomposition to Approximately Compress Convolutional Layers in Deep Learning

## Abstract
Feature extraction for tensor data serves as an important step in many tasks such as anomaly detection, process monitoring, image classification, and quality control. Although many methods have been proposed for tensor feature extraction, there are still two challenges that need to be addressed: 1) how to reduce the computation cost for high dimensional and large volume tensor data; 2) how to interpret the output features and evaluate their significance. Although the most recent methods in deep learning, such as Convolutional Neural Network (CNN), have shown outstanding performance in analyzing tensor data, their wide adoption is still hindered by model complexity and lack of interpretability. To fill this research gap, we propose to use CP-decomposition to approximately compress the convolutional layer (CPAC-Conv layer) in deep learning. The contributions of our work could be summarized into three aspects: 1) we adapt CP-decomposition to compress convolutional kernels and derive the expressions of both forward and backward propagations for our proposed CPAC-Conv layer; 2) compared with the original convolutional layer, the proposed CPAC-Conv layer can reduce the number of parameters without decaying prediction performance. It can combine with other layers to build novel Neural Networks; 3) the value of decomposed kernels indicates the significance of the corresponding feature map, which increases model interpretability and provides us insights to guide feature selection.

## Citation

To be added

## Installation

The code has been tested on following environment

```
Ubuntu 18.04
python 3.6
CUDA 10.2
tensorly 0.4.0
torch 1.4.0
scikit-learn 0.21.3
```
## Usage
There are CPU and GPU version of our model. The CPU version is limited by computational cost and only used to check the correctness of algorithm. The GPU version is built on PyTorch and used to test model performance.

### CPAC-CNN-CPU

train the CPAC-CNN on MNIST

```
python cnn_mnist.py [number of filter] [rank] [number of epoch]
```

train the CPAC-CNN on manufacturing dataset

```
python cnn_manu.py [number of filter] [rank] [number of epoch] [image size]
```

### CPAC-CNN-GPU

train the CPAC-CNN on MNIST

```
python train_mnist.py [number of filters] [filter_size] [rank] [epochs] [device]
e.g. python train_mnist.py 8 3 6 10 cuda:0
```

train the CPAC-CNN on manufacturing dataset

```
python train_manu.py [number of filters] [filter_size] [rank] [epochs] [device]
e.g. python train_manu.py 8 3 6 10 cuda:0
```

train the baseline CNN on MNIST (the "rank" is useless in this code)

```
python train_mnist_cnn.py [number of filters] [filter_size] [rank] [epochs] [device]
e.g. python train_mnist_cnn.py 8 3 6 10 cuda:0
```

train the baseline CNN on manufacturing dataset (the "rank" is useless in this code)

```
python train_manu_cnn.py [number of filters] [filter_size] [rank] [epochs] [device]
e.g. python train_manu_cnn.py 8 3 6 10 cuda:0
```

The scripts for ploting the figures in our paper are included in [folder](CPAC-CNN-GPU/result_viz/).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

The implementation of max pooling layer (maxpool.py) and fully connected layer (softmax.py) in CPAC-CNN-CPU are inspired by scripts from [Victor Zhou](https://github.com/vzhou842/cnn-from-scratch).

[MNIST](http://yann.lecun.com/exdb/mnist/)

[Magnetic Tile Defect Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.)
