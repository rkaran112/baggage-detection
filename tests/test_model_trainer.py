import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.components.model_trainer import ModelTrain


class TestToDevice(unittest.TestCase):
    def setUp(self):
        # Bypass __init__ (which downloads pretrained ResNet18 weights) since
        # _to_device only depends on self.device.
        self.trainer = ModelTrain.__new__(ModelTrain)
        self.trainer.device = torch.device("cpu")

    def test_squeezes_batch_dimension_for_normal_batch(self):
        images = torch.zeros(4, 3, 224, 224)
        targets = {"labels": torch.tensor([[0], [1], [2], [3]])}

        _, labels = self.trainer._to_device(images, targets)

        self.assertEqual(labels.shape, (4,))
        self.assertEqual(labels.tolist(), [0, 1, 2, 3])

    def test_preserves_batch_dimension_for_size_one_batch(self):
        # Regression test: squeeze() with no argument would collapse a
        # size-1 batch's labels into a 0-d scalar instead of a length-1
        # vector, breaking the loss/accuracy calculation downstream.
        images = torch.zeros(1, 3, 224, 224)
        targets = {"labels": torch.tensor([[2]])}

        _, labels = self.trainer._to_device(images, targets)

        self.assertEqual(labels.shape, (1,))
        self.assertEqual(labels.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
