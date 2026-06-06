import argparse
import random
import sys

import numpy as np
import torch

from models.backbone import build_resnet34
from models.TransSCD import TransSCD


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify whether TransSCD backbone actually uses ImageNet pretrained ResNet34."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes for TransSCD.")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Max absolute diff tolerance for exact-weight check.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        ref = build_resnet34(pretrained=True).conv1.weight.detach().cpu()
    except Exception as e:
        print("[ERROR] Failed to load torchvision ResNet34 default weights.")
        print(f"[ERROR] {type(e).__name__}: {e}")
        return 2

    try:
        net = TransSCD(in_channels=3, num_classes=args.num_classes)
        model_w = net.encoder.layer0[0].weight.detach().cpu()
    except Exception as e:
        print("[ERROR] Failed to build TransSCD model.")
        print(f"[ERROR] {type(e).__name__}: {e}")
        return 3

    max_abs_diff = torch.max(torch.abs(model_w - ref)).item()
    mean_abs_diff = torch.mean(torch.abs(model_w - ref)).item()

    print(f"model conv1 shape: {tuple(model_w.shape)}")
    print(f"ref   conv1 shape: {tuple(ref.shape)}")
    print(f"max abs diff: {max_abs_diff:.10f}")
    print(f"mean abs diff: {mean_abs_diff:.10f}")

    if max_abs_diff <= args.tol:
        print("[OK] Pretrained ResNet34 weights are loaded correctly in TransSCD.")
        return 0

    print("[FAIL] TransSCD conv1 does NOT match torchvision pretrained ResNet34.")
    print("This usually means pretrained weights were not actually loaded.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
