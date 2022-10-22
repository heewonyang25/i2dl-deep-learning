import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class FashionMNISTDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def prepare_data(self):

        # Define the transform
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Download the Fashion-MNIST dataset
        fashion_mnist_train_val = torchvision.datasets.FashionMNIST(root='../datasets', train=True,
                                                                         download=True, transform=transform)

        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../datasets', train=False,
                                                                    download=True, transform=transform)

        # Apply the Transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Perform the training and validation split
        torch.manual_seed(0)
        self.train_dataset, self.val_dataset = random_split(
            fashion_mnist_train_val, [50000, 10000])
        torch.manual_seed(torch.initial_seed())
        
    #Define the data loaders that can be called from the trainers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.fashion_mnist_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers)
