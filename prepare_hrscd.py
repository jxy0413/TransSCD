"""
Prepare HRSCD dataset for TransSCD training.

Reads the large 10000x10000 HRSCD tiles, crops them into 256x256 patches,
masks semantic labels with change labels (unchanged pixels -> class 0),
and organises everything into the SECOND-compatible directory layout:

    <output>/A/          -- 2006 image patches
    <output>/B/          -- 2012 image patches
    <output>/label1/     -- 2006 semantic label patches (masked by change)
    <output>/label2/     -- 2012 semantic label patches (masked by change)
    <output>/list/       -- train.txt / val.txt / test.txt

Usage:
    python prepare_hrscd.py --src "./datasets/HRSCD" --dst "./datasets/HRSCD_256"
"""

import os
import re
import argparse
import random
import numpy as np
import cv2
from tqdm import tqdm


def parse_coords(filename):
    """Extract the coordinate key from an HRSCD filename.

    Examples
    --------
    '14-2005-0415-6890-LA93.tif'              -> '14-0415-6890'
    '14-2012-0415-6890-LA93-0M50-E080.tif'    -> '14-0415-6890'
    '35-2006-0310-6780-LA93.tif'              -> '35-0310-6780'
    '35-2012-0310-6780-LA93-0M50-E080.tif'    -> '35-0310-6780'
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("-")
    dept = parts[0]
    coord_x = parts[2]
    coord_y = parts[3]
    return f"{dept}-{coord_x}-{coord_y}"


def build_pair_index(src_root):
    """Build a list of matched (img_A, img_B, lbl_change, lbl_A, lbl_B) paths."""
    regions = ["D14", "D35"]
    pairs = []

    for region in regions:
        dir_A = os.path.join(src_root, "images_2006", "2006", region)
        dir_B = os.path.join(src_root, "images_2012", "2012", region)
        dir_change = os.path.join(src_root, "labels_change", "change", region)
        dir_lc_A = os.path.join(src_root, "labels_land_cover_2006", "2006", region)
        dir_lc_B = os.path.join(src_root, "labels_land_cover_2012", "2012", region)

        if not os.path.isdir(dir_A):
            print(f"[WARN] Skipping {region}: {dir_A} not found")
            continue

        b_files = {parse_coords(f): f for f in os.listdir(dir_B) if f.endswith(".tif")}
        change_files = {parse_coords(f): f for f in os.listdir(dir_change) if f.endswith(".tif")}
        lc_a_files = {parse_coords(f): f for f in os.listdir(dir_lc_A) if f.endswith(".tif")}
        lc_b_files = {parse_coords(f): f for f in os.listdir(dir_lc_B) if f.endswith(".tif")}

        for fname_a in sorted(os.listdir(dir_A)):
            if not fname_a.endswith(".tif"):
                continue
            key = parse_coords(fname_a)
            if key not in b_files or key not in change_files or key not in lc_a_files or key not in lc_b_files:
                print(f"[WARN] Incomplete pair for {fname_a} (key={key}), skipping")
                continue
            pairs.append({
                "key": key,
                "img_A": os.path.join(dir_A, fname_a),
                "img_B": os.path.join(dir_B, b_files[key]),
                "lbl_change": os.path.join(dir_change, change_files[key]),
                "lbl_A": os.path.join(dir_lc_A, lc_a_files[key]),
                "lbl_B": os.path.join(dir_lc_B, lc_b_files[key]),
            })

    print(f"Found {len(pairs)} matched image pairs")
    return pairs


def crop_and_save(pairs, dst_root, patch_size=256, stride=256, min_change_ratio=0.0):
    """Crop each tile pair into patches and save."""
    os.makedirs(os.path.join(dst_root, "A"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "B"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "label1"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "label2"), exist_ok=True)

    all_ids = []
    global_idx = 0

    for pair in tqdm(pairs, desc="Processing tiles"):
        img_A = cv2.imread(pair["img_A"])
        img_B = cv2.imread(pair["img_B"])
        lbl_change = cv2.imread(pair["lbl_change"], cv2.IMREAD_UNCHANGED).astype(np.uint8)
        lbl_A = cv2.imread(pair["lbl_A"], cv2.IMREAD_UNCHANGED).astype(np.uint8)
        lbl_B = cv2.imread(pair["lbl_B"], cv2.IMREAD_UNCHANGED).astype(np.uint8)

        if img_A is None or img_B is None:
            print(f"[WARN] Failed to read images for {pair['key']}, skipping")
            continue

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # Mask semantic labels: unchanged pixels -> 0
        lbl_A_masked = lbl_A * lbl_change
        lbl_B_masked = lbl_B * lbl_change

        h, w = img_A.shape[:2]

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_change = lbl_change[y:y+patch_size, x:x+patch_size]

                change_ratio = patch_change.sum() / (patch_size * patch_size)
                if change_ratio < min_change_ratio:
                    continue

                patch_A = img_A[y:y+patch_size, x:x+patch_size]
                patch_B = img_B[y:y+patch_size, x:x+patch_size]
                patch_lbl_A = lbl_A_masked[y:y+patch_size, x:x+patch_size]
                patch_lbl_B = lbl_B_masked[y:y+patch_size, x:x+patch_size]

                pid = f"{global_idx:06d}"
                cv2.imwrite(os.path.join(dst_root, "A", pid + ".png"),
                            cv2.cvtColor(patch_A, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dst_root, "B", pid + ".png"),
                            cv2.cvtColor(patch_B, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dst_root, "label1", pid + ".png"), patch_lbl_A)
                cv2.imwrite(os.path.join(dst_root, "label2", pid + ".png"), patch_lbl_B)

                all_ids.append(pid)
                global_idx += 1

    print(f"Total patches saved: {len(all_ids)}")
    return all_ids


def split_and_write_lists(all_ids, dst_root, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train / val / test splits."""
    random.seed(seed)
    ids = list(all_ids)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train:n_train + n_val])
    test_ids = sorted(ids[n_train + n_val:])

    list_dir = os.path.join(dst_root, "list")
    os.makedirs(list_dir, exist_ok=True)

    for name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = os.path.join(list_dir, name + ".txt")
        with open(path, "w") as f:
            for pid in id_list:
                f.write(pid + "\n")
        print(f"  {name}: {len(id_list)} samples -> {path}")


def compute_mean_std(dst_root, split="train", sample_limit=2000):
    """Compute per-channel mean and std from saved patches."""
    list_path = os.path.join(dst_root, "list", split + ".txt")
    with open(list_path) as f:
        ids = [l.strip() for l in f if l.strip()]

    if len(ids) > sample_limit:
        random.seed(42)
        ids = random.sample(ids, sample_limit)

    sum_A = np.zeros(3, dtype=np.float64)
    sum_B = np.zeros(3, dtype=np.float64)
    sq_A = np.zeros(3, dtype=np.float64)
    sq_B = np.zeros(3, dtype=np.float64)
    count = 0

    for pid in tqdm(ids, desc="Computing mean/std"):
        img_a = cv2.imread(os.path.join(dst_root, "A", pid + ".png"))
        img_b = cv2.imread(os.path.join(dst_root, "B", pid + ".png"))
        if img_a is None or img_b is None:
            continue
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB).astype(np.float64)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB).astype(np.float64)

        npix = img_a.shape[0] * img_a.shape[1]
        sum_A += img_a.reshape(-1, 3).sum(axis=0)
        sum_B += img_b.reshape(-1, 3).sum(axis=0)
        sq_A += (img_a.reshape(-1, 3) ** 2).sum(axis=0)
        sq_B += (img_b.reshape(-1, 3) ** 2).sum(axis=0)
        count += npix

    mean_A = sum_A / count
    mean_B = sum_B / count
    std_A = np.sqrt(sq_A / count - mean_A ** 2)
    std_B = np.sqrt(sq_B / count - mean_B ** 2)

    print(f"\nmean_A = np.array({np.round(mean_A, 2).tolist()})")
    print(f"std_A  = np.array({np.round(std_A, 2).tolist()})")
    print(f"mean_B = np.array({np.round(mean_B, 2).tolist()})")
    print(f"std_B  = np.array({np.round(std_B, 2).tolist()})")

    return mean_A, std_A, mean_B, std_B


def main():
    parser = argparse.ArgumentParser(description="Prepare HRSCD for TransSCD training")
    parser.add_argument("--src", type=str, default="./datasets/HRSCD",
                        help="Root of the raw HRSCD dataset")
    parser.add_argument("--dst", type=str, default="./datasets/HRSCD_256",
                        help="Output directory for cropped patches")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for sliding window (default=patch_size, no overlap)")
    parser.add_argument("--min_change_ratio", type=float, default=0.0,
                        help="Min fraction of changed pixels to keep a patch (0=keep all)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_stats", action="store_true", default=True,
                        help="Compute mean/std after cropping")
    args = parser.parse_args()

    print("=" * 60)
    print("HRSCD Data Preparation")
    print("=" * 60)
    print(f"Source:     {args.src}")
    print(f"Dest:       {args.dst}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride:     {args.stride}")
    print()

    pairs = build_pair_index(args.src)
    all_ids = crop_and_save(pairs, args.dst,
                            patch_size=args.patch_size,
                            stride=args.stride,
                            min_change_ratio=args.min_change_ratio)
    split_and_write_lists(all_ids, args.dst,
                          train_ratio=args.train_ratio,
                          val_ratio=args.val_ratio,
                          seed=args.seed)

    if args.compute_stats:
        compute_mean_std(args.dst)

    print("\nDone! You can now train with:")
    print(f'  python train_SCD.py --dataname HRSCD --datapath "{os.path.abspath(args.dst)}" --num_classes 6')


if __name__ == "__main__":
    main()
