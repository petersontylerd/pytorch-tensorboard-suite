#!/bin/bash
sudo bash -c 'wget -P /home/data/mnist_raw http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
sudo bash -c 'wget -P /home/data/mnist_raw http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
sudo bash -c 'wget -P /home/data/mnist_raw http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
sudo bash -c 'wget -P /home/data/mnist_raw http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'