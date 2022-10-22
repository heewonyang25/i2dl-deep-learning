"""Tests for facial keypoint detection models"""

import os

import torch

from exercise_code.tests.base_tests import UnitTest, CompositeTest
from exercise_code.util.save_model import save_model



class ParamCountTest(UnitTest):
    """Test whether number of model params smaller than limit"""
    def __init__(self, model, limit=5e6):
        self.model = model
        self.limit = limit
        self.n_params = 0

    def test(self):
        self.n_params = sum(p.numel() for p in self.model.parameters())
        return self.n_params < self.limit

    def define_success_message(self):
        n_params_mio = self.n_params / 1e6
        return "ParamCountTest passed. Your model has {:.3f} mio. params." \
            .format(n_params_mio)

    def define_failure_message(self):
        n_params_mio = self.n_params / 1e6
        limit_mio = self.limit / 1e6
        return "Your model has {:.3f} mio. params but must have less than" \
               " {:.3f} mio. params. Simplify your model before submitting" \
               " it. You won't need that many params :)" \
            .format(n_params_mio, limit_mio)


class FileSizeTest(UnitTest):
    """Test whether file size of saved model smaller than limit"""
    def __init__(self, model, limit=20):
        self.model = model
        self.limit = limit
        self.size = 0

    def test(self):
        model_path = save_model(self.model, "model.model", ".tmp")
        size = os.path.getsize(model_path)
        self.size = size / 1e6
        return self.size < self.limit

    def define_success_message(self):
        return "FileSizeTest passed. Your model is %.1f MB large" % self.size

    def define_failure_message(self):
        return "Your model is too large! The size is {:.1f} MB, but it must" \
               " be less than {:.1f} MB. Please simplify your model before" \
               " submitting.".format(self.size, self.limit)

    def define_exception_message(self, exception):
        return "Your model could not be saved. Please make sure your model" \
               " inherits from either torch.nn.Module or" \
               " pytorch_lightning.LightningModule."


class SegModelTest(CompositeTest):
    """Composite test for SegModel"""
    def define_tests(self, model):
        return [
            ParamCountTest(model),
            FileSizeTest(model, 50)
        ]

    def define_failure_message(self):
        return "Some tests failed for your model."

    def define_success_message(self):
        return "All tests passed for your model."


def test_seg_nn(model):
    """Wrapper for SegModelTest"""
    SegModelTest(model)()
