import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from PIL import Image

from src.image_data import _resolve_image_root


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available - skipping CNN feature extraction tests")
class TestFeaturesOptional(unittest.TestCase):
    def test_extract_cnn_features(self):
        from src.features import extract_cnn_features

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for label, color in [("species_cat", "red"), ("species_dog", "blue")]:
                class_dir = root / label
                class_dir.mkdir(parents=True)
                for index in range(2):
                    image = Image.new("RGB", (128, 128), color=color)
                    image.save(class_dir / f"{label}_{index}.png")

            df = extract_cnn_features(root, image_size=64, batch_size=2, model_name="resnet18", max_images=4)
            self.assertTrue(len(df) >= 4)
            self.assertIn("label", df.columns)
            self.assertIn("file_name", df.columns)


if __name__ == "__main__":
    unittest.main()
