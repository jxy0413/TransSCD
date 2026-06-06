import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

from utils.SCD_misc import ConfuseMatrixMeter
from utils.checkpoint import load_checkpoint
from datasets import RS_ST as RS
from models.TransSCD import TransSCD as Net


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    net = Net(3, num_classes=args.num_classes).cuda()

    params_total = sum(p.numel() for p in net.parameters())
    print(f"Number of model parameters: {params_total}\n")

    test_set = RS.Data(args.datapath, 'test', num_classes=args.num_classes)
    test_loader = DataLoader(test_set, batch_size=args.test_batchsize,
                             num_workers=4, shuffle=False)
    test(args.ckptpath, test_loader, net)
    print('Testing finished.')


def test(modelpath, test_loader, net):
    N = args.num_classes
    tool4metric = ConfuseMatrixMeter(n_class=N)

    def test_phase():
        tool4metric.clear()

        ckpt = load_checkpoint(modelpath, map_location='cuda')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(test_loader, file=sys.stdout)
            for batch in loop:
                imgs_A, imgs_B, labels_A, labels_B, _, _, name = batch

                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

                out1, out2, m_out, y_tr, y_scd, _ = net(imgs_A, imgs_B)

                pred_trans = torch.argmax(y_scd, dim=1)
                preds_A = pred_trans // N
                preds_B = pred_trans % N
                same_mask = (preds_A == preds_B)
                preds_A[same_mask] = 0
                preds_B[same_mask] = 0

                pred_all = torch.cat([preds_A, preds_B], dim=0)
                label_all = torch.cat([labels_A, labels_B], dim=0)
                tool4metric.update_cm(pr=pred_all.cpu().numpy(),
                                      gt=label_all.cpu().numpy())

        scores = tool4metric.get_scores()
        print('acc={:.4f}, mIoU={:.4f}, Sek={:.4f}, Fscd={:.4f}, Pre={:.4f}, Rec={:.4f}'
              .format(scores['acc'], scores['mIoU'], scores['Sek'],
                      scores['Fscd'], scores['Pre'], scores['Rec']))

    test_phase()


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="TransSCD Testing")
    parser.add_argument("--dataname", default="SECOND", type=str)
    parser.add_argument("--modelname", default="TransSCD", type=str)
    parser.add_argument("--datapath", default="", type=str, help="data path")
    parser.add_argument("--ckptpath", default="", type=str, help="checkpoint path")
    parser.add_argument("--vispath", default="", type=str, help="visualization path")
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--test_batchsize', type=int, default=1)
    args = parser.parse_args()

    if args.dataname in RS.DATASET_CONFIGS:
        RS.set_dataset_config(args.dataname)

    if args.vispath == "":
        args.vispath = os.path.join(working_path, "results")

    vispath = os.path.join(args.vispath, args.dataname)
    for sub in ["labelA_rgb", "labelB_rgb"]:
        p = os.path.join(vispath, sub)
        if not os.path.exists(p):
            os.makedirs(p)

    main(args)
