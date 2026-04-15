"""
Prepare the JL1 dataset for TransSCD training.

What this script does:
  1. Generate list files (train.txt / val.txt / test.txt) inside each split
     directory so that the Data loader can find them directly.
  2. Compute per-time (im1 / im2) channel mean and std for train images and
     print the values for pasting into DATASET_CONFIGS in RS_ST.py.

Usage:
    python prepare_jl1.py --root "./datasets/JL1"
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def collect_ids(directory):
    """Return sorted list of stem names (no extension) for all images."""
    names = [os.path.splitext(f)[0] for f in os.listdir(directory)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    return sorted(names)


def write_list(out_path, ids):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for name in ids:
            f.write(name + "\n")
    print(f"  Wrote {len(ids)} entries -> {out_path}")


def compute_mean_std(img_dir, ids, ext):
    """Incremental mean / std over all images (per-channel, pixel-level)."""
    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)

    for name in tqdm(ids, desc=f"  Stats for {os.path.basename(img_dir)}"):
        img = np.array(Image.open(os.path.join(img_dir, name + ext)))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, :3].astype(np.float64)
        npx = img.shape[0] * img.shape[1]
        n_pixels += npx
        channel_sum += img.reshape(-1, 3).sum(axis=0)
        channel_sq_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    return mean, std


def detect_ext(directory):
    exts = {}
    for f in os.listdir(directory):
        _, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext in IMG_EXTS:
            exts[ext] = exts.get(ext, 0) + 1
    if not exts:
        return ".png"
    return max(exts, key=exts.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root of JL1 dataset (contains train/val/test)")
    args = parser.parse_args()
    root = args.root

    splits = {
        "train": os.path.join(root, "train"),
        "val":   os.path.join(root, "val"),
        "test":  os.path.join(root, "test"),
    }

    print("=" * 60)
    print("Step 1: Generate list files")
    print("=" * 60)

    for split_name, split_dir in splits.items():
        if not os.path.isdir(split_dir):
            print(f"  [skip] {split_dir} not found")
            continue
        im1_dir = os.path.join(split_dir, "im1")
        if not os.path.isdir(im1_dir):
            print(f"  [skip] {im1_dir} not found")
            continue
        ids = collect_ids(im1_dir)
        list_path = os.path.join(split_dir, "list", split_name + ".txt")
        write_list(list_path, ids)

    print()
    print("=" * 60)
    print("Step 2: Compute mean / std on train split")
    print("=" * 60)

    train_dir = splits["train"]
    im1_dir = os.path.join(train_dir, "im1")
    im2_dir = os.path.join(train_dir, "im2")
    ext = detect_ext(im1_dir)
    ids = collect_ids(im1_dir)

    mean_A, std_A = compute_mean_std(im1_dir, ids, ext)
    mean_B, std_B = compute_mean_std(im2_dir, ids, ext)

    def fmt(arr):
        return "[" + ", ".join(f"{v:.2f}" for v in arr) + "]"

    print()
    print("Paste the following into DATASET_CONFIGS['JL1H'] in datasets/RS_ST.py:")
    print(f'  "mean_A": np.array({fmt(mean_A)}),')
    print(f'  "std_A":  np.array({fmt(std_A)}),')
    print(f'  "mean_B": np.array({fmt(mean_B)}),')
    print(f'  "std_B":  np.array({fmt(std_B)}),')
    print()
    print("Done!")


if __name__ == "__main__":
    main()
