#!/usr/bin python3.7
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# pytorch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def load_mnist(path, kind='train'):
    """
    Load MNIST data from 'path'
    """
    labels_path = os.path.join(path,
                               '{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path,
                               '{}-images-idx3-ubyte'.format(kind))
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
                             len(labels), 784)
        images = ((images / 255.) - .5) * 2
    
    return images, labels


class ISICDatasetTrain(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target


def image_sample(inp, title=None, figsize=(20,20)):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=figsize)
    plt.imshow(inp, interpolation='nearest')
    if title is not None:
        plt.title(title)
    plt.savefig("./test_2.jpg")


if __name__ == "__main__":
    
    mnist_dir=os.path.join(os.environ["HOME"], "s3buckets", "mnist")

    ## training
    # images
    X_train, y_train = load_mnist(
        path=mnist_dir,
        kind="train"
    )

    train_data = ISICDatasetTrain(
        images=X_train,
        targets=y_train,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=16,
        shuffle=False,
        # sampler=weighted_sampler
    )
    # print(type(train_data_loader))

    # visualize image batch grid
    inputs, classes = next(iter(train_data_loader))
    out = torchvision.utils.make_grid(inputs)

    image_sample(out)