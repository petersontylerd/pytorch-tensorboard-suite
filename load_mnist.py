import os
import numpy as np
import struct

def load_mnist(path, kind="train"):
    """
    Documentation:

        ---
        Description:
            Load MNIST images and labels from unzipped source files.

        ---
        Parameters:
            kind : str
                Used to identify training data vs. validation data. Pass
                "train" to load training data, and "t10k" to load validation
                data.

        ---
        Returns
            images : Numpy array
                Numpy array containing all images in dataset. Has shape N by
                784, where N is the number of samples and 784 is the number
                of pixels.
            targets : Numpy array
                Numpy array containing all targets associated with images.
                Has shape N by 1, where N is the number of samples.
    """

    labels_path = os.path.join(path,
                               "{}-labels-idx1-ubyte".format(kind))
    images_path = os.path.join(path,
                               "{}-images-idx3-ubyte".format(kind))

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II",
                                 lbpath.read(8))
        targets = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
                             len(targets), 784)

    return images, targets
