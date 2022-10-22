"""Tests for ImageFolderDataset in data/image_folder_dataset.py"""

import random

from .base_tests import UnitTest, MethodTest, ClassTest, test_results_to_score
from .len_tests import LenTest
import numpy as np


class GetItemTestType(UnitTest):
    """Test whether __getitem()__ returns correct data type"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.type = dict
        self.wrong_type = None

    def test(self):
        random_indices = random.sample(range(len(self.dataset)), 100)
        for i in random_indices:
            if not isinstance(self.dataset[i], self.type):
                self.wrong_type = type(self.dataset[i])
                return False
        return True

    def define_failure_message(self):
        return "Expected __getitem()__ to return type %s but got %s." \
               % (self.type, self.wrong_type)


class GetItemTestImageShape(UnitTest):
    """Test whether images loaded by __getitem__() are of correct shape"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.expected_shape = (32, 32, 3)
        self.wrong_shape = None

    def test(self):
        random_indices = random.sample(range(len(self.dataset)), 100)
        for i in random_indices:
            if self.dataset[i]["image"].shape != self.expected_shape:
                self.wrong_shape = self.dataset[i]["image"].shape
                return False
        return True

    def define_failure_message(self):
        return "Expected images to have shape %s but got %s." \
               % (str(self.expected_shape), str(self.dataset.images.shape))

    
class GetItemTestTransformApplied(UnitTest):
    """Test whether images loaded by __getitem__() are of correct shape"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        # Set transform
        self.dataset.transform = transform
        self.transform = transform
        self.dataset_entries = None
        self.loaded_entries = None

    def test(self):
        random_indices = random.sample(range(len(self.dataset)), 100)
        for i in random_indices:
            # Load images manually
            image = self.dataset.load_image_as_numpy(self.dataset.images[i])
            transformed_image = self.transform(image)
            dataset_image = self.dataset[i]["image"]
            if not np.array_equal(dataset_image, transformed_image):
                self.dataset_entries = dataset_image[0][0]
                self.loaded_entries = transformed_image[0][0]
                return False
        return True

    def define_failure_message(self):
        return "Expected images to be successfully transformed but input image" \
               "first entries are %s and should be %s." \
               % (str(self.dataset_entries), str(self.loaded_entries))


class GetItemTest(MethodTest):
    """Test __getitem__() method of ImageFolderDataset"""
    def define_tests(self, dataset):
        return [
            GetItemTestType(dataset),
            GetItemTestImageShape(dataset),
        ]

    def define_method_name(self):
        return "__getitem__"


class ImageFolderDatasetTest(ClassTest):
    """Test class ImageFolderDataset"""
    def define_tests(self, dataset):
        return [
            LenTest(dataset, 50000),
            GetItemTest(dataset),
        ]

    def define_class_name(self):
        return "ImageFolderDataset"

    
class ImageFolderTransformTest(MethodTest):
    """Test class ImageFolderDataset"""
    def define_tests(self, dataset):
        return [
            GetItemTestType(dataset),
            GetItemTestImageShape(dataset),
            GetItemTestTransformApplied(dataset, lambda x: x*2),
        ]
    def define_method_name(self):
        return "__getitem__"


def test_image_folder_dataset(dataset):
    """Test class ImageFolderDataset"""
    test = ImageFolderDatasetTest(dataset)
    return test_results_to_score(test())


def test_transform_dataset(dataset):
    test = ImageFolderTransformTest(dataset)
    return test_results_to_score(test())


## Implementations for testing of __len__ and __getitem__ functions seperately

def test_len_dataset(dataset):
    """Test method LenTest"""
    test = LenTest(dataset, 50000)
    return test_results_to_score(test())

def test_item_dataset(dataset):
    """Test method GetItemTest"""
    test = GetItemTest(dataset)
    return test_results_to_score(test())

