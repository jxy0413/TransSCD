import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

ST_COLORMAP = np.asarray(
    [
        [255, 255, 255],  # unchanged
        [0, 128, 0],      # low vegetation
        [128, 128, 128],  # ground
        [0, 255, 0],      # tree
        [0, 0, 255],      # water
        [128, 0, 0],      # building
        [255, 0, 0],      # sports field
    ],
    dtype=np.uint8,
)


def read_split_ids(list_file: Path) -> List[str]:
    with list_file.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def validate_required_dirs(root: Path) -> Tuple[List[str], Dict[str, Path]]:
    required = {
        "A": root / "A",
        "B": root / "B",
        "label1": root / "label1",
        "label2": root / "label2",
        "list": root / "list",
    }
    errors = []
    for name, p in required.items():
        if not p.exists() or not p.is_dir():
            errors.append(f"Missing required directory: {p} ({name})")
    return errors, required


def check_split_lists(list_dir: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    split_files = {
        "train": list_dir / "train.txt",
        "val": list_dir / "val.txt",
        "test": list_dir / "test.txt",
    }
    errors: List[str] = []
    splits: Dict[str, List[str]] = {}

    for split, file_path in split_files.items():
        if not file_path.exists():
            errors.append(f"Missing split file: {file_path}")
            continue
        ids = read_split_ids(file_path)
        if not ids:
            errors.append(f"Split file is empty: {file_path}")
            continue
        splits[split] = ids
    return errors, splits


def check_split_overlap(splits: Dict[str, List[str]]) -> List[str]:
    warns: List[str] = []
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a_name = split_names[i]
            b_name = split_names[j]
            a_set = set(splits[a_name])
            b_set = set(splits[b_name])
            inter = a_set.intersection(b_set)
            if inter:
                warns.append(
                    f"{a_name}.txt and {b_name}.txt overlap: {len(inter)} samples "
                    f"(example: {next(iter(inter))})"
                )
    return warns


def check_file_existence(root: Path, splits: Dict[str, List[str]], suffix: str) -> List[str]:
    errors: List[str] = []
    subdirs = ["A", "B", "label1", "label2"]

    all_ids: Set[str] = set()
    for split_ids in splits.values():
        all_ids.update(split_ids)

    for sample_id in sorted(all_ids):
        for sub in subdirs:
            p = root / sub / f"{sample_id}{suffix}"
            if not p.exists():
                errors.append(f"Missing file: {p}")
    return errors


def load_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to read label image: {path}")
    # cv2 loads color images as BGR; SECOND colormap is defined in RGB.
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def color_to_index(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    if arr.ndim == 2:
        return arr.astype(np.uint8), 0

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported label shape: {arr.shape}")

    idx = np.zeros(arr.shape[:2], dtype=np.uint8)
    matched = np.zeros(arr.shape[:2], dtype=bool)
    for cls_idx, color in enumerate(ST_COLORMAP):
        mask = np.all(arr == color, axis=-1)
        idx[mask] = cls_idx
        matched |= mask

    unknown = int((~matched).sum())
    return idx, unknown


def check_label_values(
    root: Path,
    splits: Dict[str, List[str]],
    suffix: str,
    num_classes: int,
    max_scan: int,
) -> Tuple[List[str], List[str], Dict[str, Set[int]]]:
    errors: List[str] = []
    warns: List[str] = []
    stats = {"label1": set(), "label2": set()}
    color_label_count = {"label1": 0, "label2": 0}
    unknown_color_pixels = {"label1": 0, "label2": 0}

    all_ids: List[str] = []
    seen = set()
    for split in ["train", "val", "test"]:
        for sample_id in splits.get(split, []):
            if sample_id not in seen:
                seen.add(sample_id)
                all_ids.append(sample_id)

    scan_ids = all_ids if max_scan <= 0 else all_ids[:max_scan]
    allowed = set(range(num_classes))

    for sample_id in scan_ids:
        for label_dir in ["label1", "label2"]:
            p = root / label_dir / f"{sample_id}{suffix}"
            if not p.exists():
                continue
            try:
                arr = load_label(p)
            except ValueError as e:
                errors.append(str(e))
                continue

            if arr.ndim == 3:
                color_label_count[label_dir] += 1
                try:
                    arr, unknown = color_to_index(arr)
                except ValueError as e:
                    errors.append(f"{p}: {e}")
                    continue
                unknown_color_pixels[label_dir] += unknown

            unique_values = np.unique(arr)
            stats[label_dir].update(int(v) for v in unique_values.tolist())
            bad_values = [int(v) for v in unique_values.tolist() if int(v) not in allowed]
            if bad_values:
                errors.append(
                    f"Out-of-range labels in {p}: {sorted(set(bad_values))} "
                    f"(expected in [0, {num_classes - 1}])"
                )

    if max_scan > 0 and len(all_ids) > max_scan:
        warns.append(
            f"Only scanned first {max_scan}/{len(all_ids)} unique IDs for label values. "
            "Use --max_scan 0 for full scan."
        )

    for label_dir in ["label1", "label2"]:
        if color_label_count[label_dir] > 0:
            warns.append(
                f"{label_dir}: detected {color_label_count[label_dir]} RGB color labels "
                "(this is acceptable for SECOND; checker mapped them to class IDs)."
            )
        if unknown_color_pixels[label_dir] > 0:
            errors.append(
                f"{label_dir}: found {unknown_color_pixels[label_dir]} pixels with colors "
                "outside SECOND colormap."
            )

    return errors, warns, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SECOND dataset consistency for TransSCD training.")
    parser.add_argument(
        "--datapath",
        type=str,
        default="datasets/SECOND_train_set",
        help="Path to dataset root.",
    )
    parser.add_argument("--num_classes", type=int, default=7, help="Expected number of classes.")
    parser.add_argument("--suffix", type=str, default=".png", help="Image filename suffix.")
    parser.add_argument(
        "--max_scan",
        type=int,
        default=500,
        help="How many unique IDs to scan for label values. 0 means full scan.",
    )
    args = parser.parse_args()

    root = Path(args.datapath)
    print(f"Checking dataset: {root.resolve()}")
    print(f"Expected classes: [0, {args.num_classes - 1}]")
    print(f"Filename suffix: {args.suffix}")

    errors: List[str] = []
    warns: List[str] = []

    dir_errors, required = validate_required_dirs(root)
    errors.extend(dir_errors)
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        print("\nCheck failed.")
        return 1

    list_errors, splits = check_split_lists(required["list"])
    errors.extend(list_errors)
    if not errors:
        for split_name in ["train", "val", "test"]:
            if split_name in splits:
                print(f"[OK] {split_name}.txt: {len(splits[split_name])} entries")

    warns.extend(check_split_overlap(splits))

    file_errors = check_file_existence(root, splits, args.suffix)
    errors.extend(file_errors)
    if not file_errors:
        print("[OK] All listed images and labels exist.")

    label_errors, label_warns, label_stats = check_label_values(
        root=root,
        splits=splits,
        suffix=args.suffix,
        num_classes=args.num_classes,
        max_scan=args.max_scan,
    )
    errors.extend(label_errors)
    warns.extend(label_warns)

    for label_dir in ["label1", "label2"]:
        if label_stats[label_dir]:
            values = sorted(label_stats[label_dir])
            print(f"[INFO] {label_dir} observed labels: {values}")

    if warns:
        print("\nWarnings:")
        for w in warns:
            print(f"[WARN] {w}")

    if errors:
        print("\nErrors:")
        max_show = 30
        for e in errors[:max_show]:
            print(f"[ERROR] {e}")
        if len(errors) > max_show:
            print(f"[ERROR] ... and {len(errors) - max_show} more errors")
        print(f"\nCheck failed with {len(errors)} error(s).")
        return 1

    print("\nCheck passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
