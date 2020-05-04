#!/usr/bin python3.7
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    
    mnist_dir=os.path.join(os.environ["HOME"], "s3buckets", "mnist")

    ## training
    # images
    X_train, y_train = load_mnist(
        path=mnist_dir,
        kind="train"
    )
    # print(len(images))
    # print(len(labels))

    fig, ax = plt.subplots(nrows=2, ncols=5,
                        sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("./test.jpg")