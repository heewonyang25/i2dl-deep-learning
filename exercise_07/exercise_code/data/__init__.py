"""Definition of all datasets and dataloader"""

from .base_dataset import DummyDataset
from .image_folder_dataset import (
    ImageFolderDataset,
    MemoryImageFolderDataset,
)
from .dataloader import DataLoader
