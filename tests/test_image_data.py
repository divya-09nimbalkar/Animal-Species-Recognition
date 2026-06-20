import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.image_data import discover_image_dataset, load_image_dataset


class TestImageData(unittest.TestCase):
    def test_load_image_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for label, color in [("species_cat", "red"), ("species_dog", "blue")]:
                class_dir = root / label
                class_dir.mkdir(parents=True)
                for index in range(2):
                    image = Image.new("RGB", (64, 64), color=color)
                    image.save(class_dir / f"{label}_{index}.png")

            self.assertTrue(discover_image_dataset(root))
            dataset = load_image_dataset(root, image_size=32)
            self.assertEqual(len(dataset), 4)
            self.assertIn("label", dataset.columns)
            self.assertIn("pixel_0", dataset.columns)


if __name__ == "__main__":
    unittest.main()
