"""
Download sample cat/dog images from a public torchvision dataset.
Run this if you don't have your own images yet.
"""
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

DATA_DIR = Path("data/train")
SOURCE_DIR = Path("data/source")
MAX_PER_CLASS = 500  # Increase for more training data


def download():
    print("Downloading Oxford-IIIT Pet dataset via torchvision...")
    ds = OxfordIIITPet(
        root=str(SOURCE_DIR),
        split="trainval",
        target_types="binary-category",  # 0 = cat, 1 = dog
        download=True,
    )

    for split in ["cats", "dogs"]:
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)

    counts = {"cats": 0, "dogs": 0}
    label_map = {0: "cats", 1: "dogs"}

    for img, label in ds:
        cls = label_map[int(label)]
        if counts[cls] >= MAX_PER_CLASS:
            continue
        out_path = DATA_DIR / cls / f"{cls}_{counts[cls]:05d}.jpg"
        img.save(out_path)
        counts[cls] += 1
        if counts["cats"] >= MAX_PER_CLASS and counts["dogs"] >= MAX_PER_CLASS:
            break

    print(f"Downloaded {counts['cats']} cats and {counts['dogs']} dogs into {DATA_DIR}")


if __name__ == "__main__":
    download()
