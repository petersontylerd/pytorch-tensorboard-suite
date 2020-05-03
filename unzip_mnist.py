import gzip
import os
import shutil

def unzip_gzip_file(gzip_dir, unzip_dir, kind="train"):

    gzip_labels_path = os.path.join(gzip_dir, '{}-labels-idx1-ubyte.gz'.format(kind))
    gzip_images_path = os.path.join(gzip_dir, '{}-images-idx3-ubyte.gz'.format(kind))
    
    unzip_labels_path = os.path.join(unzip_dir, '{}-labels-idx1-ubyte'.format(kind))
    unzip_images_path = os.path.join(unzip_dir, '{}-images-idx3-ubyte'.format(kind))

    with gzip.open(gzip_labels_path, "rb") as file_in:
        with open(unzip_labels_path, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)
    
    with gzip.open(gzip_images_path, "rb") as file_in:
        with open(unzip_images_path, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


if __name__ == "__main__":

    unzip_gzip_file(
        gzip_dir="/home/data/mnist_raw",
        unzip_dir="/home/data/mnist_unzip",
    )

    