import os
import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from utils.loss import TransSCDLoss
from utils.SCD_misc import ConfuseMatrixMeter, AverageMeter
from utils.checkpoint import load_checkpoint

from datasets import RS_ST as RS

from models.TransSCD import TransSCD as Net


def cosine_one_cycle_lr(optimizer, current_step, total_steps, init_lr,
                        warmup_steps=0, min_lr=1e-6):
    if current_step < warmup_steps:
        lr = init_lr * current_step / max(warmup_steps, 1)
    else:
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    writer = SummaryWriter(args.chkpt_dir)

    net = Net(3, num_classes=args.num_classes).cuda()

    params_total = sum(p.numel() for p in net.parameters())
    print(f"Number of model parameters: {params_total}\n")

    train_set = RS.Data(args.train_datapath, 'train', augmentation=True,
                        num_classes=args.num_classes)
    train_loader = DataLoader(train_set, batch_size=args.train_batchsize,
                              num_workers=4, shuffle=True)
    val_set = RS.Data(args.val_datapath, args.val_mode,
                      num_classes=args.num_classes)
    val_loader = DataLoader(val_set, batch_size=args.val_batchsize,
                            num_workers=4, shuffle=False)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = TransSCDLoss(
        num_classes=args.num_classes,
        w_ss=args.w_ss, w_prior=args.w_prior,
        w_tr=args.w_tr, w_cons=args.w_cons, w_scd=args.w_scd,
    )

    start_epoch = 0
    bestscore = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        ckpt = load_checkpoint(args.resume, map_location='cuda')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            bestscore = ckpt.get('bestscore', 0)
            print(f"=> Resumed from epoch {start_epoch}, bestscore={bestscore:.4f}")
        else:
            net.load_state_dict(ckpt)
            import re
            m = re.search(r'E(\d+)', os.path.basename(args.resume))
            if m:
                start_epoch = int(m.group(1)) + 1
            print(f"=> Loaded model weights, resuming from epoch {start_epoch}")

    train(train_loader, val_loader, net, optimizer, criterion, writer,
          start_epoch, bestscore)
    writer.close()
    print('Training finished.')


def train(train_loader, val_loader, net, optimizer, criterion, writer,
          start_epoch=0, bestscore=0):
    N = args.num_classes
    tool4metric = ConfuseMatrixMeter(n_class=N)

    total_steps = len(train_loader) * args.epoch
    accum_steps = args.accum_steps

    def training_phase(epc):
        torch.cuda.empty_cache()
        net.train()

        meters = {k: AverageMeter() for k in
                  ['total', 'ss', 'prior', 'tr', 'cons', 'scd']}

        global_step_base = epc * len(train_loader)
        optimizer.zero_grad()

        loop = tqdm(train_loader, file=sys.stdout)
        for i, batch in enumerate(loop):
            loop.set_description(f'Epoch:{epc}')

            imgs_A, imgs_B, labels_A, labels_B, change_label, trans_label, _ = batch

            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()
            change_label = change_label.cuda().float()
            trans_label = trans_label.cuda().long()

            step = global_step_base + i
            cosine_one_cycle_lr(optimizer, step, total_steps, args.lr)

            out1, out2, m_out, y_tr, y_scd, cons_loss = net(
                imgs_A, imgs_B, labels_A, labels_B
            )

            loss, loss_dict = criterion(
                out1, out2, m_out, y_tr, y_scd, cons_loss,
                labels_A, labels_B, change_label, trans_label,
            )

            (loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            for k in meters:
                meters[k].update(loss_dict[k].item() if torch.is_tensor(loss_dict[k])
                                 else loss_dict[k])

            loop.set_postfix(loss=meters['total'].val,
                             lr=optimizer.param_groups[0]['lr'])

        msg = 'LOSS %.4f [ss %.4f prior %.4f tr %.4f cons %.4f scd %.4f]' % (
            meters['total'].val, meters['ss'].val, meters['prior'].val,
            meters['tr'].val, meters['cons'].val, meters['scd'].val)
        print(msg)

        writer.add_scalar('train/total_loss', meters['total'].val, epc)
        writer.add_scalar('train/ss_loss', meters['ss'].val, epc)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epc)

    def validation_phase(epc):
        tool4metric.clear()
        net.eval()
        torch.cuda.empty_cache()

        val_loss = AverageMeter()

        with torch.no_grad():
            loop = tqdm(val_loader, file=sys.stdout)
            for batch in loop:
                loop.set_description(f'Val:{epc}')

                imgs_A, imgs_B, labels_A, labels_B, change_label, trans_label, _ = batch

                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

                out1, out2, m_out, y_tr, y_scd, _ = net(imgs_A, imgs_B)

                loss_A = nn.CrossEntropyLoss(ignore_index=0)(out1, labels_A)
                loss_B = nn.CrossEntropyLoss(ignore_index=0)(out2, labels_B)
                val_loss.update((0.5 * (loss_A + loss_B)).item())

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
        print('acc={:.4f}, mIoU={:.4f}, Sek={:.4f}, Fscd={:.4f}'.format(
            scores['acc'], scores['mIoU'], scores['Sek'], scores['Fscd']))

        writer.add_scalar('val/loss', val_loss.average(), epc)
        writer.add_scalar('val/mIoU', scores['mIoU'], epc)
        writer.add_scalar('val/Sek', scores['Sek'], epc)
        return scores

    for epc in range(start_epoch, args.epoch):
        training_phase(epc)
        score = validation_phase(epc)

        if score['Sek'] > bestscore:
            bestscore = score['Sek']
            torch.save({
                'epoch': epc,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bestscore': bestscore,
            }, os.path.join(args.chkpt_dir,
                            "E{}_iou{:.2f}_Sek{:.2f}.pth".format(
                                epc, score['mIoU'] * 100, score['Sek'] * 100)))


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="TransSCD Training")
    parser.add_argument("--dataname", default="SECOND", type=str)
    parser.add_argument("--modelname", default="TransSCD", type=str)
    parser.add_argument("--datapath", default="", type=str, help="data path")
    parser.add_argument("--train_datapath", default="", type=str)
    parser.add_argument("--val_datapath", default="", type=str)
    parser.add_argument("--val_mode", default="val", type=str)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--lr', type=float, default=3.5e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train_batchsize', type=int, default=2)
    parser.add_argument('--val_batchsize', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=8,
                        help='gradient accumulation steps (effective batch=batchsize*accum)')
    parser.add_argument('--w_ss', type=float, default=1.0, help='semantic loss weight')
    parser.add_argument('--w_prior', type=float, default=1.0, help='change prior loss weight')
    parser.add_argument('--w_tr', type=float, default=1.0, help='transition loss weight')
    parser.add_argument('--w_cons', type=float, default=0.1, help='consistency loss weight')
    parser.add_argument('--w_scd', type=float, default=1.0, help='SCD loss weight')
    parser.add_argument('--resume', type=str, default="")
    args = parser.parse_args()

    if args.dataname in RS.DATASET_CONFIGS:
        RS.set_dataset_config(args.dataname)

    if args.train_datapath == "":
        args.train_datapath = args.datapath
    if args.val_datapath == "":
        args.val_datapath = args.datapath

    chkpt_dir = os.path.join(working_path, 'checkpoints', args.dataname, args.modelname)
    pred_dir = os.path.join(working_path, 'results', args.dataname)

    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    run_dir = sorted([f for f in os.listdir(chkpt_dir) if f.startswith("run_")])

    if args.resume and len(run_dir) > 0:
        num_run = int(run_dir[-1].split("_")[-1])
    elif len(run_dir) > 0:
        num_run = int(run_dir[-1].split("_")[-1]) + 1
    else:
        num_run = 0

    args.chkpt_dir = os.path.join(chkpt_dir, "run_%04d" % num_run + "/")
    args.pred_dir = pred_dir

    main(args)
