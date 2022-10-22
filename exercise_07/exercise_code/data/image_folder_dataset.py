"""
Definition of ImageFolderDataset dataset class
and image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import os
import pickle
import numpy as np
from PIL import Image, ImageOps
import random

from .base_dataset import Dataset


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""
    def __init__(self, *args, transform=None, mode='train', limit_files=None,
                 split={'train': 0.6, 'val': 0.2, 'test': 0.2},
                 download_url="https://cdn3.vision.in.tum.de/~dl4cv/cifar10.zip",
                 **kwargs):
        super().__init__(*args, 
                         download_url=download_url,
                         **kwargs)
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0
        
        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.split = split
        self.limit_files = limit_files
        self.images, self.labels = self.make_dataset(
            directory=self.root_path,
            class_to_idx=self.class_to_idx,
            mode=mode,
        )
        self.transform = transform

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def select_split(self, images, labels, mode):
        """
        Depending on the mode of the dataset, deterministically split it.
        
        :param images, a list containing paths to all images in the dataset
        :param labels, a list containing one label per image
        
        :returns (images, labels), where only the indices for the
            corresponding data split are selected.
        """
        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(images)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)
        
        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)
        
        if mode == 'train':
            idx = rand_perm[:num_train]
        elif mode == 'val':
            idx = rand_perm[num_train:num_train+num_valid]
        elif mode == 'test':
            idx = rand_perm[num_train+num_valid:]

        if self.limit_files:
            idx = idx[:self.limit_files]
        
        if isinstance(images, list): 
            return list(np.array(images)[idx]), list(np.array(labels)[idx])
        else: 
            return images[idx], list(np.array(labels)[idx])

    def make_dataset(self, directory, class_to_idx, mode):
        """
        Create the image dataset by preparaing a list of samples
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset
            - labels is a list containing one label per image
        """
        images, labels = [], []
        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append(path)
                    labels.append(label)

        images, labels = self.select_split(images, labels, mode)

        assert len(images) == len(labels)
        return images, labels
        
    def __len__(self):
        length = None
        length = len(self.images)
        return length

    @staticmethod
    def load_image_as_pil_image(image_path):
        """Load image from image_path as numpy array"""
        return Image.open(image_path)

    def __getitem__(self, index):
        data_dict = None

        label = self.labels[index]
        path = self.images[index]
        image = self.load_image_as_pil_image(path)
        if self.transform is not None:
            image = self.transform(image)
        data_dict = {
            "image": image,
            "label": label,
        }
        # Make it like a pytorch ImageFolderDataset output
        data_dict = [image, label]

        return data_dict
    
class MemoryImageFolderDataset(ImageFolderDataset):
    def __init__(self, root, *args,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/cifar10memory.zip",
                 **kwargs):
        # Fix the root directory automatically
        if not root.endswith('memory'):
            root += 'memory'

        super().__init__(
            root, *args, transform=transform, download_url=download_url, **kwargs)

    @staticmethod
    def _find_classes(directory):
        with open(os.path.join(
            directory, 'cifar10.pckl'
            ), 'rb') as f:
            save_dict = pickle.load(f)
        class_to_idx = save_dict['class_to_idx'] # dict
        classes = save_dict['classes'] # list
        
        return classes, class_to_idx
    def make_dataset(self, directory, class_to_idx, mode):
        with open(os.path.join(
            directory, 'cifar10.pckl'
            ), 'rb') as f:
            save_dict = pickle.load(f)

        images = save_dict['images'] # np array (50000, 32,32, 3)
        labels = save_dict['labels'] # list of 50000 elements

        images, labels = self.select_split(images, labels, mode)

        assert len(images) == len(labels)
        return images, labels
    def load_image_as_pil_image(self, image_path):
        """Here we already have everything in memory,
        so we can just return the image"""
        return Image.fromarray(image_path.astype('uint8'))
