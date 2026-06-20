"""Generate a small sample image dataset for demos and tests."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"
root.mkdir(parents=True, exist_ok=True)

for label, color in [("species_cat", "red"), ("species_dog", "blue")]:
    d = root / label
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        img = Image.new("RGB", (128, 128), color=color)
        img.save(d / f"{label}_{i}.png")

print("Created sample images at:", root)
