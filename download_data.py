"""
Download cat/dog images from Open Images (dataset-backed, reliable).
Targets are copied into:
  - data/train/cats
  - data/train/dogs
"""
from argparse import ArgumentParser
import shutil
from pathlib import Path

from openimages.download import download_dataset

DATA_DIR = Path("data/train")
SOURCE_DIR = Path("data/source/openimages")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_images(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def sync_from_source(class_name: str, source_label: str, target_count: int) -> int:
    target_dir = DATA_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    source_images = SOURCE_DIR / source_label / "images"

    existing = count_images(target_dir)
    if existing >= target_count:
        return existing
    if not source_images.exists():
        return existing

    index = existing
    for src in sorted(source_images.iterdir()):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            continue
        if index >= target_count:
            break
        dst = target_dir / f"{class_name}_{index:05d}{src.suffix.lower()}"
        if dst.exists():
            index += 1
            continue
        shutil.copy2(src, dst)
        index += 1

    return count_images(target_dir)


def download_openimages(source_label: str, target_count: int):
    csv_dir = SOURCE_DIR / "csv"
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OpenImages '{source_label}' up to {target_count} images...")
    download_dataset(
        dest_dir=str(SOURCE_DIR),
        class_labels=[source_label],
        csv_dir=str(csv_dir),
        limit=target_count,
    )


def parse_args():
    parser = ArgumentParser(description="Download cat/dog image dataset")
    parser.add_argument("--cats", type=int, default=20000, help="Target number of cat images")
    parser.add_argument("--dogs", type=int, default=20000, help="Target number of dog images")
    return parser.parse_args()


def main():
    args = parse_args()

    download_openimages("Cat", args.cats)
    download_openimages("Dog", args.dogs)

    cats = sync_from_source("cats", "Cat", args.cats)
    dogs = sync_from_source("dogs", "Dog", args.dogs)
    print(f"Done. cats={cats}, dogs={dogs}, target=({args.cats}, {args.dogs})")


if __name__ == "__main__":
    main()
