import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage import io
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from datasets import RS_ST as RS
from models.TransSCD import TransSCD as Net
from utils.checkpoint import load_checkpoint


def Index2Color(pred):
    colormap = np.asarray(RS.ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def denormalize(img, time='A'):
    if time == 'A':
        img = img * RS.STD_A + RS.MEAN_A
    else:
        img = img * RS.STD_B + RS.MEAN_B
    return np.clip(img, 0, 255).astype(np.uint8)


def visualize_single(img_A, img_B, label_A, label_B,
                     pred_A, pred_B, change_prior, name, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'TransSCD - {name}', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img_A)
    axes[0, 0].set_title('Image T1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_B)
    axes[0, 1].set_title('Image T2')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(change_prior, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Soft Change Prior')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(Index2Color(label_A))
    axes[1, 0].set_title('GT T1')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(Index2Color(label_B))
    axes[1, 1].set_title('GT T2')
    axes[1, 1].axis('off')

    gt_change = ((label_A > 0) | (label_B > 0)).astype(np.uint8)
    axes[1, 2].imshow(gt_change, cmap='gray')
    axes[1, 2].set_title('GT Change')
    axes[1, 2].axis('off')

    axes[2, 0].imshow(Index2Color(pred_A))
    axes[2, 0].set_title('Pred T1')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(Index2Color(pred_B))
    axes[2, 1].set_title('Pred T2')
    axes[2, 1].axis('off')

    legend_elements = []
    for color, cls_name in zip(RS.ST_COLORMAP, RS.ST_CLASSES):
        color_norm = np.array(color) / 255.0
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color_norm, label=cls_name))
    axes[2, 2].legend(handles=legend_elements, loc='center', fontsize=10)
    axes[2, 2].set_title('Legend')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{name}_vis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    N = args.num_classes
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_path = os.path.join(args.save_path, 'visualizations')
    os.makedirs(save_path, exist_ok=True)

    print(f"Loading model from: {args.ckptpath}")
    net = Net(3, num_classes=N).cuda()

    ckpt = load_checkpoint(args.ckptpath, map_location='cuda')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    test_set = RS.Data(args.datapath, 'test', num_classes=N)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

    print(f"Total test samples: {len(test_set)}")
    print(f"Visualizing {min(args.num_samples, len(test_set))} samples...")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= args.num_samples:
                break

            imgs_A, imgs_B, labels_A, labels_B, _, _, name = batch
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()

            out1, out2, m_out, y_tr, y_scd, _ = net(imgs_A, imgs_B)

            pred_trans = torch.argmax(y_scd, dim=1)
            preds_A = (pred_trans // N).cpu().numpy()[0]
            preds_B = (pred_trans % N).cpu().numpy()[0]
            same_mask = (preds_A == preds_B)
            preds_A[same_mask] = 0
            preds_B[same_mask] = 0

            img_A_np = imgs_A[0].cpu().numpy().transpose(1, 2, 0)
            img_B_np = imgs_B[0].cpu().numpy().transpose(1, 2, 0)
            img_A_np = denormalize(img_A_np, 'A')
            img_B_np = denormalize(img_B_np, 'B')

            change_prior = m_out[0].cpu().numpy()

            sample_name = name[0] if isinstance(name, (list, tuple)) else name

            visualize_single(
                img_A_np, img_B_np,
                labels_A[0].numpy(), labels_B[0].numpy(),
                preds_A, preds_B, change_prior,
                sample_name, save_path
            )

            pred_A_color = Index2Color(preds_A)
            pred_B_color = Index2Color(preds_B)
            io.imsave(os.path.join(save_path, f'{sample_name}_pred_T1.png'), pred_A_color)
            io.imsave(os.path.join(save_path, f'{sample_name}_pred_T2.png'), pred_B_color)

            print(f"  [{idx+1}] Processed: {sample_name}")

    print(f"\nDone! Results saved to: {save_path}")


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="TransSCD Visualization")
    parser.add_argument("--dataname", default="SECOND", type=str)
    parser.add_argument("--datapath", default="", type=str)
    parser.add_argument("--ckptpath", default="", type=str)
    parser.add_argument("--save_path", default="./results/SECOND", type=str)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.dataname in RS.DATASET_CONFIGS:
        RS.set_dataset_config(args.dataname)

    main(args)
