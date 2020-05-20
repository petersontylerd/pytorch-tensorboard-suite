#!/bin/bash
sudo wget -P $HOME/data/mnist_raw http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
sudo wget -P $HOME/data/mnist_raw http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
sudo wget -P $HOME/data/mnist_raw http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
sudo wget -P $HOME/data/mnist_raw http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

python3.7 $HOME/workspace/pytorch-tensorboard-suite/unzip_mnist.py