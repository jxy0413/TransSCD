"""
Compute per-channel mean and std for JL1 training images.

Run once, then paste the printed values into DATASET_CONFIGS["JL1H"]
in datasets/RS_ST.py to replace the placeholder statistics.

Usage:
    python compute_jl1_stats.py --root "path/to/JL1/train"
"""

import os
import numpy as np
from PIL import Image


def compute_mean_std(img_dir):
    IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in os.listdir(img_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)

    n_pixels = 0
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq = np.zeros(3, dtype=np.float64)

    total = len(files)
    for i, fname in enumerate(files):
        img = np.array(Image.open(os.path.join(img_dir, fname)))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, :3].astype(np.float64)
        npx = img.shape[0] * img.shape[1]
        n_pixels += npx
        ch_sum += img.reshape(-1, 3).sum(axis=0)
        ch_sq += (img.reshape(-1, 3) ** 2).sum(axis=0)
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {os.path.basename(img_dir)}")

    mean = ch_sum / n_pixels
    std = np.sqrt(ch_sq / n_pixels - mean ** 2)
    return mean, std


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets/JL1/train",
                        help="Root of JL1 training split (contains im1/, im2/)")
    cli_args = parser.parse_args()
    train_root = cli_args.root

    print("Computing im1 (time A) statistics ...")
    mean_A, std_A = compute_mean_std(os.path.join(train_root, "im1"))

    print("Computing im2 (time B) statistics ...")
    mean_B, std_B = compute_mean_std(os.path.join(train_root, "im2"))

    fmt = lambda a: "[" + ", ".join(f"{v:.2f}" for v in a) + "]"
    print()
    print("=" * 60)
    print("Paste into DATASET_CONFIGS['JL1H'] in datasets/RS_ST.py:")
    print("=" * 60)
    print(f'        "mean_A": np.array({fmt(mean_A)}),')
    print(f'        "std_A":  np.array({fmt(std_A)}),')
    print(f'        "mean_B": np.array({fmt(mean_B)}),')
    print(f'        "std_B":  np.array({fmt(std_B)}),')
    print()


if __name__ == "__main__":
    main()
