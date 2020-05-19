#!/usr/bin python3.7
import gzip
import os
import shutil


def unzip_gzip_file(gzip_dir, unzip_dir, kind="train"):

    if not os.path.exists(os.path.dirname(unzip_dir)):
        os.makedirs(os.path.dirname(unzip_dir), exist_ok=True)

    with gzip.open(gzip_dir, "rb") as file_in:
        with open(unzip_dir, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


if __name__ == "__main__":

    ## training
    # images
    unzip_gzip_file(
        gzip_dir=os.path.join(os.environ["HOME"], "data", "mnist_raw", 'train-images-idx3-ubyte.gz'),
        unzip_dir=os.path.join(os.environ["HOME"], "s3buckets", "mnist", 'train-images-idx3-ubyte'),
    )
    # labels
    unzip_gzip_file(
        gzip_dir=os.path.join(os.environ["HOME"], "data", "mnist_raw", 'train-labels-idx1-ubyte.gz'),
        unzip_dir=os.path.join(os.environ["HOME"], "s3buckets", "mnist", 'train-labels-idx1-ubyte'),
    )

    ## test
    # images
    unzip_gzip_file(
        gzip_dir=os.path.join(os.environ["HOME"], "data", "mnist_raw", 't10k-images-idx3-ubyte.gz'),
        unzip_dir=os.path.join(os.environ["HOME"], "s3buckets", "mnist", 't10k-images-idx3-ubyte'),
    )
    # labels
    unzip_gzip_file(
        gzip_dir=os.path.join(os.environ["HOME"], "data", "mnist_raw", 't10k-labels-idx1-ubyte.gz'),
        unzip_dir=os.path.join(os.environ["HOME"], "s3buckets", "mnist", 't10k-labels-idx1-ubyte'),
    )

