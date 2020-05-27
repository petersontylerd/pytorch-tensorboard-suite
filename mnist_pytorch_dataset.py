from torch.utils.data import Dataset, DataLoader, TensorDataset

class MNISTDataset(Dataset):
    """
    Documentation:

        ---
        Description:
            Load MNIST images and labels into a Pytorch Dataset.

        ---
        Parameters:
            images : Numpy array
                Numpy array containing all images in dataset. Has shape N by
                784, where N is the number of samples and 784 is the number
                of pixels.
            targets : Numpy array
                Numpy array containing all targets associated with images.
                Has shape N by 1, where N is the number of samples.
            transform : Pytorch transforms object
                Optional transformation instructions for images
    """
    def __init__(self, images, targets, transform=None):
        self.images = images.reshape(-1,28,28)
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target