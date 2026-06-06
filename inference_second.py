import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from datasets import RS_ST as RS
from models.TransSCD import TransSCD as Net
from utils.checkpoint import load_checkpoint


def Index2Color(pred):
    colormap = np.asarray(RS.ST_COLORMAP, dtype='uint8')
    return colormap[np.asarray(pred, dtype='int32'), :]


def main(args):
    N = args.num_classes
    pred_A_path = os.path.join(args.save_path, 'pred_T1_rgb')
    pred_B_path = os.path.join(args.save_path, 'pred_T2_rgb')
    os.makedirs(pred_A_path, exist_ok=True)
    os.makedirs(pred_B_path, exist_ok=True)

    print(f"Loading model: {args.ckptpath}")
    net = Net(3, num_classes=N).cuda()

    ckpt = load_checkpoint(args.ckptpath, map_location='cuda')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    test_set = RS.Data(args.datapath, 'test', num_classes=N)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    print(f"Total samples: {len(test_set)}")

    with torch.no_grad():
        for imgs_A, imgs_B, labels_A, labels_B, _, _, name in tqdm(test_loader, desc="Inference"):
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()

            out1, out2, m_out, y_tr, y_scd, _ = net(imgs_A, imgs_B)

            pred_trans = torch.argmax(y_scd, dim=1)
            preds_A = (pred_trans // N).cpu().numpy()[0]
            preds_B = (pred_trans % N).cpu().numpy()[0]
            same_mask = (preds_A == preds_B)
            preds_A[same_mask] = 0
            preds_B[same_mask] = 0

            sample_name = name[0] if isinstance(name, (list, tuple)) else name

            io.imsave(os.path.join(pred_A_path, f'{sample_name}.png'), Index2Color(preds_A))
            io.imsave(os.path.join(pred_B_path, f'{sample_name}.png'), Index2Color(preds_B))

    print(f"\nDone! Results saved to:")
    print(f"  T1: {pred_A_path}")
    print(f"  T2: {pred_B_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", default="SECOND", type=str)
    parser.add_argument("--datapath", default="", type=str)
    parser.add_argument("--ckptpath", default="", type=str)
    parser.add_argument("--save_path", default="./results/SECOND", type=str)
    parser.add_argument('--num_classes', type=int, default=7)
    args = parser.parse_args()

    if args.dataname in RS.DATASET_CONFIGS:
        RS.set_dataset_config(args.dataname)

    main(args)
