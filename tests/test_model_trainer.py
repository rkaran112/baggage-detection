import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.components.model_trainer import ModelTrain


class FakeModel(nn.Module):
    # Returns pre-canned logits per call instead of doing a real forward
    # pass, so test_model's accuracy computation can be checked without
    # loading an actual ResNet18.
    def __init__(self, outputs_per_call):
        super().__init__()
        self.outputs_per_call = outputs_per_call
        self.calls = 0

    def forward(self, x):
        out = self.outputs_per_call[self.calls]
        self.calls += 1
        return out


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


class TestTestModel(unittest.TestCase):
    def setUp(self):
        # Bypass __init__ (which downloads pretrained ResNet18 weights) and
        # supply a fake model whose predictions are known in advance, so the
        # accuracy computation can be checked without a real forward pass.
        self.trainer = ModelTrain.__new__(ModelTrain)
        self.trainer.device = torch.device("cpu")

    def test_accuracy_matches_fraction_of_correct_predictions(self):
        batch1_targets = {"labels": torch.tensor([[0], [1]])}
        batch1_outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])  # both correct

        batch2_targets = {"labels": torch.tensor([[0], [1]])}
        batch2_outputs = torch.tensor([[10.0, 0.0], [10.0, 0.0]])  # one wrong

        self.trainer.test_loader = [
            (torch.zeros(2, 3, 224, 224), batch1_targets),
            (torch.zeros(2, 3, 224, 224), batch2_targets),
        ]
        self.trainer.model = FakeModel([batch1_outputs, batch2_outputs])

        accuracy = self.trainer.test_model()

        self.assertAlmostEqual(accuracy, 75.0)


if __name__ == "__main__":
    unittest.main()
