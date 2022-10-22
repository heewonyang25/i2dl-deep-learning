"""
Definition of ImageFolderDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import pickle

import numpy as np
from PIL import Image

from .base_dataset import Dataset


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""
    def __init__(self, *args,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/cifar10.zip",
                 **kwargs):
        super().__init__(*args, 
                         download_url=download_url,
                         **kwargs)
        
        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.images, self.labels = self.make_dataset(
            directory=self.root_path,
            class_to_idx=self.class_to_idx
        )
        # transform function that we will apply later for data preprocessing
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

    @staticmethod
    def make_dataset(directory, class_to_idx):
        """
        Create the image dataset by preparaing a list of samples
        Images are sorted in an ascending order by class and file name
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset, NOT the actual images
            - labels is a list containing one label per image
        """
        images, labels = [], []

        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset (number of images)                  #
        ########################################################################

        length = len(self.images)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length

    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
        return np.asarray(Image.open(image_path), dtype=float)

    def __getitem__(self, index):
        data_dict = None
        ########################################################################
        # TODO:                                                                #
        # create a dict of the data at the given index in your dataset         #
        # The dict should be of the following format:                          #
        # {"image": <i-th image>,                                              #
        # "label": <label of i-th image>}                                      #
        # Hints:                                                               #
        #   - use load_image_as_numpy() to load an image from a file path      #
        #   - If applicable (Task 4: 'Transforms and Image Preprocessing'),    #
        #     make sure to apply self.transform to the image if one is defined:#                           
        #     image_transformed = self.transform(image)                        #
        ########################################################################

        data_dict = {}
        image = self.load_image_as_numpy(self.images[index])

        if self.transform is None:
            data_dict['image'] = image
            data_dict['label'] = self.labels[index]
        else:
            data_dict['image'] = (self.transform(image))
            data_dict['label'] = self.labels[index]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
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
            root, *args, download_url=download_url, **kwargs)
        
        with open(os.path.join(
            self.root_path, 'cifar10.pckl'
            ), 'rb') as f:
            save_dict = pickle.load(f)

        self.images = save_dict['images']
        self.labels = save_dict['labels']
        self.class_to_idx = save_dict['class_to_idx']
        self.classes = save_dict['classes']

        self.transform = transform

    def load_image_as_numpy(self, image_path):
        """Here we already have everything in memory,
        so we can just return the image"""
        return image_path

        