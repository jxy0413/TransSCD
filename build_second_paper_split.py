import argparse
import random
from pathlib import Path
from typing import List, Set


def read_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in ids:
            f.write(f"{x}\n")


def dedup_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def infer_ids_from_dir(root: Path) -> List[str]:
    for sub in ["A", "im1"]:
        d = root / sub
        if d.exists() and d.is_dir():
            ids = sorted([p.stem for p in d.glob("*.png")])
            if ids:
                return ids
    raise FileNotFoundError(
        f"Could not infer IDs from {root}. Expected one of: {root / 'A'} or {root / 'im1'}"
    )


def backup_if_exists(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] Backup: {bak}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build SECOND split following TransSCD paper protocol: "
            "keep original train set unchanged, split original test set into val/test."
        )
    )
    parser.add_argument(
        "--src_train_list",
        type=str,
        default="",
        help="Path to official original SECOND train list (expected ~2968 IDs).",
    )
    parser.add_argument(
        "--src_test_list",
        type=str,
        default="",
        help="Path to official original SECOND test list (expected ~1694 IDs).",
    )
    parser.add_argument(
        "--src_train_root",
        type=str,
        default="",
        help="Fallback: infer train IDs from <root>/A/*.png or <root>/im1/*.png.",
    )
    parser.add_argument(
        "--src_test_root",
        type=str,
        default="",
        help="Fallback: infer test IDs from <root>/A/*.png or <root>/im1/*.png.",
    )
    parser.add_argument(
        "--out_train_list_dir",
        type=str,
        default="datasets/SECOND_train_set/list",
        help="Output directory for train.txt (2968 IDs).",
    )
    parser.add_argument(
        "--out_eval_list_dir",
        type=str,
        default="datasets/SECOND_test_test/test/list",
        help="Output directory for val.txt / test.txt (847/847 from original test).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_size", type=int, default=847, help="Validation set size.")
    parser.add_argument("--test_size", type=int, default=847, help="Testing set size.")
    parser.add_argument(
        "--strict_paper_counts",
        action="store_true",
        help="Fail if source list sizes are not exactly train=2968 and test=1694.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print split stats without writing files.",
    )
    args = parser.parse_args()

    out_train_dir = Path(args.out_train_list_dir)
    out_eval_dir = Path(args.out_eval_list_dir)
    train_ids: List[str]
    test_orig_ids: List[str]

    if args.src_train_list:
        src_train = Path(args.src_train_list)
        if not src_train.exists():
            print(f"[ERROR] Missing --src_train_list: {src_train}")
            return 1
        train_ids = dedup_keep_order(read_ids(src_train))
    elif args.src_train_root:
        src_train_root = Path(args.src_train_root)
        try:
            train_ids = dedup_keep_order(infer_ids_from_dir(src_train_root))
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return 1
    else:
        print("[ERROR] Please provide --src_train_list or --src_train_root.")
        return 1

    if args.src_test_list:
        src_test = Path(args.src_test_list)
        if not src_test.exists():
            print(f"[ERROR] Missing --src_test_list: {src_test}")
            return 1
        test_orig_ids = dedup_keep_order(read_ids(src_test))
    elif args.src_test_root:
        src_test_root = Path(args.src_test_root)
        try:
            test_orig_ids = dedup_keep_order(infer_ids_from_dir(src_test_root))
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return 1
    else:
        print("[ERROR] Please provide --src_test_list or --src_test_root.")
        return 1

    print(f"Source train IDs: {len(train_ids)}")
    print(f"Source test  IDs: {len(test_orig_ids)}")

    if args.strict_paper_counts:
        if len(train_ids) != 2968 or len(test_orig_ids) != 1694:
            print(
                "[ERROR] strict_paper_counts enabled, but sizes are not "
                f"train=2968/test=1694. Got train={len(train_ids)}, test={len(test_orig_ids)}."
            )
            return 1

    if args.val_size + args.test_size != len(test_orig_ids):
        print(
            f"[ERROR] val_size + test_size must equal original test size. "
            f"Got {args.val_size} + {args.test_size} != {len(test_orig_ids)}."
        )
        return 1

    overlap = set(train_ids).intersection(set(test_orig_ids))
    if overlap:
        print(
            f"[WARN] Source train/test overlap detected: {len(overlap)} IDs "
            f"(example: {next(iter(overlap))})."
        )

    rng = random.Random(args.seed)
    shuffled_test = test_orig_ids.copy()
    rng.shuffle(shuffled_test)

    val_ids = sorted(shuffled_test[: args.val_size])
    test_ids = sorted(shuffled_test[args.val_size : args.val_size + args.test_size])

    assert len(set(val_ids).intersection(set(test_ids))) == 0

    print(f"Output train size: {len(train_ids)} (unchanged)")
    print(f"Output val size:   {len(val_ids)}")
    print(f"Output test size:  {len(test_ids)}")
    print(f"Output val/test overlap: {len(set(val_ids).intersection(set(test_ids)))}")
    print(f"Seed: {args.seed}")

    train_out = out_train_dir / "train.txt"
    val_out = out_eval_dir / "val.txt"
    test_out = out_eval_dir / "test.txt"

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return 0

    backup_if_exists(train_out)
    backup_if_exists(val_out)
    backup_if_exists(test_out)

    write_ids(train_out, train_ids)
    write_ids(val_out, val_ids)
    write_ids(test_out, test_ids)

    print(f"[OK] Wrote: {train_out}")
    print(f"[OK] Wrote: {val_out}")
    print(f"[OK] Wrote: {test_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

