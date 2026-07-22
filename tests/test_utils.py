import os
import sys
import unittest

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed for tests
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import visualize_sample


class FakeDataset:
    def __init__(self, image, target):
        self.image = image
        self.target = target

    def __getitem__(self, index):
        return self.image, self.target


class TestVisualizeSample(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_box_scaled_from_normalized_yolo_coords_to_pixels(self):
        # 100 (H) x 200 (W) image; box is normalized YOLO format (0-1 range)
        # centered at (0.5, 0.5) covering 20% of each dimension.
        image = torch.zeros(3, 100, 200)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        dataset = FakeDataset(image, {"boxes": boxes})

        visualize_sample(dataset, 0)

        patches = plt.gca().patches
        self.assertEqual(len(patches), 1)

        rect = patches[0]
        self.assertAlmostEqual(rect.get_width(), 40.0)  # 0.2 * 200
        self.assertAlmostEqual(rect.get_height(), 20.0)  # 0.2 * 100
        self.assertAlmostEqual(rect.get_x(), 80.0)  # 0.5*200 - 40/2
        self.assertAlmostEqual(rect.get_y(), 40.0)  # 0.5*100 - 20/2

    def test_no_boxes_draws_no_rectangles(self):
        image = torch.zeros(3, 50, 50)
        boxes = torch.empty((0, 4))
        dataset = FakeDataset(image, {"boxes": boxes})

        visualize_sample(dataset, 0)

        self.assertEqual(len(plt.gca().patches), 0)


if __name__ == "__main__":
    unittest.main()
