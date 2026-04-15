import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-8, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        b, n, _, _ = predict.shape
        target = make_one_hot(target.unsqueeze(1), n).cuda()
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class Logit_Interaction_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, out_A, out_B, out_change):
        logit_A = torch.softmax(out_A, dim=1)
        logit_B = torch.softmax(out_B, dim=1)
        logit_Change = torch.softmax(out_change, dim=1)

        pred_A = torch.argmax(logit_A, dim=1)
        pred_B = torch.argmax(logit_B, dim=1)

        unchange = pred_A == pred_B
        change = ~unchange

        CD_logit = logit_Change[:, 0, :, :] * unchange + logit_Change[:, 1, :, :] * change

        max_A, _ = torch.max(logit_A, dim=1)
        max_B, _ = torch.max(logit_B, dim=1)
        SCD_logit = 0.5 * (max_A + max_B)

        loss = self.mse(SCD_logit, CD_logit.detach())
        return loss


# ======================== TransSCD Losses (Eq. 12-14) ========================

class TransSCDLoss(nn.Module):
    """
    Combined loss for TransSCD (Eq. 14):
    L = λ_ss * L_ss + λ_prior * L_prior + λ_tr * L_tr + λ_cons * L_cons + λ_scd * L_scd
    """
    def __init__(self, num_classes=7,
                 w_ss=1.0, w_prior=1.0, w_tr=1.0, w_cons=0.1, w_scd=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.w_ss = w_ss
        self.w_prior = w_prior
        self.w_tr = w_tr
        self.w_cons = w_cons
        self.w_scd = w_scd

        self.ce_sem = nn.CrossEntropyLoss(ignore_index=0)
        self.bce_prior = nn.BCELoss()
        self.ce_tr = nn.CrossEntropyLoss()
        self.ce_scd = nn.CrossEntropyLoss()

    def forward(self, out1, out2, m_out, y_tr, y_scd, consistency_loss,
                label_A, label_B, change_label, trans_label):
        loss_ss = 0.5 * (self.ce_sem(out1, label_A) + self.ce_sem(out2, label_B))

        loss_prior = self.bce_prior(m_out, change_label)

        loss_tr = self.ce_tr(y_tr, trans_label)

        loss_scd = self.ce_scd(y_scd, trans_label)

        total = (self.w_ss * loss_ss +
                 self.w_prior * loss_prior +
                 self.w_tr * loss_tr +
                 self.w_cons * consistency_loss +
                 self.w_scd * loss_scd)

        loss_dict = {
            'total': total,
            'ss': loss_ss,
            'prior': loss_prior,
            'tr': loss_tr,
            'cons': consistency_loss,
            'scd': loss_scd,
        }
        return total, loss_dict
