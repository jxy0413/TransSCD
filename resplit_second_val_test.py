import argparse
import random
from pathlib import Path
from typing import List, Set, Tuple


def read_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_ids(path: Path, ids: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak


def dedup_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_eval_pool(train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> List[str]:
    train_set = set(train_ids)
    eval_union = dedup_keep_order(val_ids + test_ids)
    pool = [x for x in eval_union if x not in train_set]
    return pool


def split_pool(pool: List[str], val_size: int, test_size: int, seed: int) -> Tuple[List[str], List[str]]:
    if val_size + test_size > len(pool):
        raise ValueError(
            f"Requested val_size + test_size = {val_size + test_size}, "
            f"but only {len(pool)} samples available in eval pool."
        )

    rng = random.Random(seed)
    work = pool.copy()
    rng.shuffle(work)

    new_val = sorted(work[:val_size])
    new_test = sorted(work[val_size:val_size + test_size])
    return new_val, new_test


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resplit SECOND val/test into mutually exclusive sets with reproducible seed."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="datasets/SECOND_train_set",
        help="Path to dataset root containing list/train.txt, list/val.txt, list/test.txt",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val_size",
        type=int,
        default=-1,
        help="New val size. Default: half of eval pool.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=-1,
        help="New test size. Default: remaining half of eval pool.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned split stats; do not write files.",
    )
    args = parser.parse_args()

    list_dir = Path(args.datapath) / "list"
    train_p = list_dir / "train.txt"
    val_p = list_dir / "val.txt"
    test_p = list_dir / "test.txt"

    for p in [train_p, val_p, test_p]:
        if not p.exists():
            print(f"[ERROR] Missing file: {p}")
            return 1

    train_ids = dedup_keep_order(read_ids(train_p))
    val_ids = dedup_keep_order(read_ids(val_p))
    test_ids = dedup_keep_order(read_ids(test_p))

    old_inter = set(val_ids).intersection(set(test_ids))
    print(f"Old split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Old val/test overlap: {len(old_inter)}")

    pool = build_eval_pool(train_ids, val_ids, test_ids)
    print(f"Eval pool (excluding train): {len(pool)}")

    if args.val_size < 0 and args.test_size < 0:
        val_size = len(pool) // 2
        test_size = len(pool) - val_size
    elif args.val_size >= 0 and args.test_size >= 0:
        val_size = args.val_size
        test_size = args.test_size
    else:
        print("[ERROR] Please set both --val_size and --test_size, or set neither.")
        return 1

    try:
        new_val, new_test = split_pool(pool, val_size, test_size, args.seed)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    new_inter = set(new_val).intersection(set(new_test))
    print(f"New split sizes: val={len(new_val)}, test={len(new_test)}")
    print(f"New val/test overlap: {len(new_inter)}")
    print(f"Seed: {args.seed}")

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return 0

    val_bak = backup(val_p)
    test_bak = backup(test_p)
    write_ids(val_p, new_val)
    write_ids(test_p, new_test)

    print(f"[OK] Wrote: {val_p}")
    print(f"[OK] Wrote: {test_p}")
    print(f"[OK] Backup: {val_bak}")
    print(f"[OK] Backup: {test_bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

