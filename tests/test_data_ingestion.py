import importlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image

from src.components.data_ingestion import CustomDataset
import src.components.data_ingestion as data_ingestion


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.images_dir = os.path.join(self.tmp_dir.name, "images")
        self.labels_dir = os.path.join(self.tmp_dir.name, "labels")
        os.makedirs(self.images_dir)
        os.makedirs(self.labels_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write_sample(self, name, label_contents):
        Image.new("RGB", (10, 10)).save(os.path.join(self.images_dir, f"{name}.jpg"))
        with open(os.path.join(self.labels_dir, f"{name}.txt"), "w") as f:
            f.write(label_contents)

    def test_getitem_parses_box_and_label(self):
        self._write_sample("a", "1 0.5 0.5 0.2 0.2\n")
        dataset = CustomDataset(self.images_dir, self.labels_dir)

        _, target = dataset[0]

        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(
            [round(v, 2) for v in target["boxes"][0].tolist()], [0.5, 0.5, 0.2, 0.2]
        )

    def test_getitem_skips_blank_lines_in_label_file(self):
        # Trailing/blank lines are common in real YOLO-format label files
        # produced by annotation tools.
        self._write_sample("a", "1 0.5 0.5 0.2 0.2\n\n")
        dataset = CustomDataset(self.images_dir, self.labels_dir)

        _, target = dataset[0]

        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(len(target["boxes"]), 1)

    def test_len_matches_number_of_samples(self):
        self._write_sample("a", "0 0.5 0.5 0.2 0.2\n")
        self._write_sample("b", "1 0.5 0.5 0.2 0.2\n")
        dataset = CustomDataset(self.images_dir, self.labels_dir)

        self.assertEqual(len(dataset), 2)


class TestDataIngestionConfig(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("BAGGAGE_DATA_DIR", None)
        importlib.reload(data_ingestion)

    def test_data_dir_defaults_when_env_var_unset(self):
        os.environ.pop("BAGGAGE_DATA_DIR", None)
        importlib.reload(data_ingestion)

        self.assertEqual(
            data_ingestion.DataIngestionConfig.data_dir,
            "D:/Code/ML_projects/baggage_detection/data",
        )

    def test_data_dir_honors_env_var_override(self):
        os.environ["BAGGAGE_DATA_DIR"] = "/tmp/some_dataset"
        importlib.reload(data_ingestion)

        self.assertEqual(data_ingestion.DataIngestionConfig.data_dir, "/tmp/some_dataset")
        self.assertEqual(
            data_ingestion.DataIngestionConfig.train_images_dir,
            os.path.join("/tmp/some_dataset", "train", "images"),
        )


if __name__ == "__main__":
    unittest.main()
