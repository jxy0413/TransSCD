import argparse
from pathlib import Path
from typing import List, Set


def read_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_list_file(root: Path, filename: str) -> Path:
    candidate = root / "list" / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing expected list file: {candidate}")


def infer_ids_from_image_dir(root: Path) -> List[str]:
    # Support common SECOND folder styles:
    # - .../A/*.png
    # - .../im1/*.png
    for sub in ["A", "im1"]:
        d = root / sub
        if d.exists() and d.is_dir():
            ids = sorted([p.stem for p in d.glob("*.png")])
            if ids:
                return ids
    raise FileNotFoundError(
        f"Could not infer IDs from {root}. Expected one of: {root / 'A'} or {root / 'im1'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether SECOND train/test sets match expected official counts and are disjoint."
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="datasets/SECOND_train_set",
        help="Root folder of SECOND train set (contains list/train.txt).",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        required=True,
        help="Root folder of SECOND test set (contains list/test.txt).",
    )
    parser.add_argument("--strict", action="store_true", help="Enforce train=2968, test=1694.")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    try:
        train_list = find_list_file(train_root, "train.txt")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    # Some test packages ship IDs as list/test.txt, others as list/all.txt, and
    # some have no list at all (infer from im1/A filenames).
    test_list = test_root / "list" / "test.txt"
    test_ids: List[str]
    if test_list.exists():
        test_ids = read_ids(test_list)
        test_source = str(test_list)
    else:
        alt = test_root / "list" / "all.txt"
        if alt.exists():
            test_ids = read_ids(alt)
            test_source = str(alt)
        else:
            try:
                test_ids = infer_ids_from_image_dir(test_root)
                test_source = f"{test_root}/(A|im1) inferred"
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                return 1

    train_ids = read_ids(train_list)
    train_set: Set[str] = set(train_ids)
    test_set: Set[str] = set(test_ids)

    print(f"train list: {train_list}")
    print(f"test  IDs source: {test_source}")
    print(f"train IDs: {len(train_ids)} (unique={len(train_set)})")
    print(f"test  IDs: {len(test_ids)} (unique={len(test_set)})")

    overlap = train_set.intersection(test_set)
    print(f"train/test overlap: {len(overlap)}")
    if overlap:
        sample = next(iter(overlap))
        print(f"[WARN] Example overlap ID: {sample}")

    if args.strict:
        ok = True
        if len(train_set) != 2968:
            print(f"[ERROR] train unique count != 2968 (got {len(train_set)})")
            ok = False
        if len(test_set) != 1694:
            print(f"[ERROR] test unique count != 1694 (got {len(test_set)})")
            ok = False
        if overlap:
            print("[ERROR] train/test must be disjoint.")
            ok = False
        if not ok:
            return 2

    print("[OK] Check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

