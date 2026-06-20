import tempfile
import unittest

from src.main import main


class TestMainEntryPoint(unittest.TestCase):
    def test_run_train_mode_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main(
                [
                    "--mode",
                    "train",
                    "--data-dir",
                    "data/raw",
                    "--output-dir",
                    tmpdir,
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                ]
            )
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
