import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )
    def forward(self, x):
        return self.dwconv(x)


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class CBA3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ECA(nn.Module):
    def __init__(self, kernal=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernal, padding=(kernal - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class task_interaction_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')

    def forward(self, old_sem1, old_sem2, old_change, change_result):
        sem_max_out, _ = torch.max(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_avg_out = torch.mean(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_out = self.sigmoid(self.Sem2Change(torch.cat([sem_max_out, sem_avg_out], dim=1)))
        new_change = old_change * sem_out

        b, c, h, w = old_sem1.size()
        fea_sem1 = torch.reshape(old_sem1.permute(0,2,3,1), [b*h*w, c])
        fea_sem2 = torch.reshape(old_sem2.permute(0,2,3,1), [b*h*w, c])

        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        target = unchange_mask.float()
        target = target - change_mask.float()
        target = torch.reshape(target, [b * h * w])
        similarity_loss = self.loss_f(fea_sem1, fea_sem2, target)
        return new_change, similarity_loss


class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #upsample
        self.upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.catconv = CBA3x3(out_channel * 2, out_channel)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, up, skip):
        #upsample
        up = self.upconv(up)
        up = torch.cat([up, skip], dim=1)
        up = self.catconv(up)
        return up

# BCFE
class Change_Specific_Transfer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = CBA1x1(in_channel*2, in_channel)
        self.eca = ECA()
        self.resblock = self._make_layer(ResBlock, 256, 128, 6, stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        xc1 = self.conv(torch.cat([x1, x2], dim=1))
        xc2 = self.conv(torch.cat([x2, x1], dim=1))
        change = self.eca(xc1 + xc2)
        diff = torch.abs(x1 - x2)
        change = torch.cat([change, diff], dim=1)
        change = self.resblock(change)
        return change


class Multi_Level_Feature_Aggregation(nn.Module):

    def __init__(self,):
        super(Multi_Level_Feature_Aggregation, self).__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)

        self.cat_conv = CBA1x1(384, 128)


    def forward(self, x1, x2, x3):
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.cat_conv(x)
        return x


class Boundary_Decoder(nn.Module):
    def __init__(self):
        super(Boundary_Decoder, self).__init__()
        self.sobel_x, self.sobel_y = get_sobel(64, 1)
        self.conv = nn.Conv2d(64, 64, 1, 1, 0)

    def forward(self, x, size):
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x = run_sobel(self.sobel_x, self.sobel_y, x)
        x = self.conv(x)
        return x


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


# ======================== TransSCD Modules ========================

class SemanticDecoder(nn.Module):
    """Shared semantic decoder: upsample high-level features and fuse with skip."""
    def __init__(self, in_ch=128, out_ch=64):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=7, stride=2,
                                     padding=3, output_padding=1)
        self.fuse = CBA3x3(out_ch * 2, out_ch)

    def forward(self, x_high, x_low):
        x_high = self.up(x_high)
        x = torch.cat([x_high, x_low], dim=1)
        return self.fuse(x)


class SoftChangePrior(nn.Module):
    """Eq.(1): m = sigma(f_prior(|S1 - S2|))"""
    def __init__(self, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, s1, s2):
        diff = torch.abs(s1 - s2)
        return self.head(diff)


class TransitionQueryConstructor(nn.Module):
    """Eq.(2): q_i = f_q([S1, S2, |S1-S2|, S2-S1])"""
    def __init__(self, feat_dim=128, query_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim * 4, query_dim * 2, 1),
            nn.BatchNorm2d(query_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim * 2, query_dim, 1),
        )

    def forward(self, s1, s2):
        cat = torch.cat([s1, s2, torch.abs(s1 - s2), s2 - s1], dim=1)
        return self.proj(cat)


class TransitionPrototypeReasoning(nn.Module):
    """
    Prototype bank (Eq.3), soft-prior guided matching (Eq.4-6),
    transition-enhanced feature aggregation (Eq.7),
    and region-level consistency loss (Eq.9-10).
    """
    def __init__(self, num_classes=7, query_dim=128,
                 tau=0.07, prior_lambda=1.0, margin=0.5):
        super().__init__()
        self.N = num_classes
        self.N2 = num_classes * num_classes
        self.query_dim = query_dim
        self.tau = tau
        self.lam = prior_lambda
        self.margin = margin

        self.prototypes = nn.Parameter(torch.randn(self.N2, query_dim) * 0.02)

    def forward(self, q_flat, m_flat, B, H, W, label_A=None, label_B=None):
        """
        q_flat: (BHW, d)
        m_flat: (BHW,)  soft change prior per pixel
        """
        q_norm = F.normalize(q_flat, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        sim = torch.mm(q_norm, p_norm.t()) / self.tau  # (BHW, N2)

        eps = 1e-6
        log_m = torch.log(m_flat.unsqueeze(1) + eps)
        log_1m = torch.log(1.0 - m_flat.unsqueeze(1) + eps)

        change_mask_proto = torch.ones(self.N2, device=q_flat.device)
        for a in range(self.N):
            change_mask_proto[a * self.N + a] = 0.0

        prior_bias = (change_mask_proto.unsqueeze(0) * self.lam * log_m +
                      (1.0 - change_mask_proto.unsqueeze(0)) * self.lam * log_1m)
        sim_adjusted = sim + prior_bias

        alpha = F.softmax(sim_adjusted, dim=-1)  # (BHW, N2)

        r = torch.mm(alpha, self.prototypes)  # (BHW, d)

        consistency_loss = torch.tensor(0.0, device=q_flat.device)
        if label_A is not None and label_B is not None:
            consistency_loss = self._consistency_loss(
                q_flat, r, label_A, label_B, B, H, W
            )

        return r, consistency_loss

    def _consistency_loss(self, q_flat, r_flat, label_A, label_B, B, H, W):
        feat_h, feat_w = H, W
        la = F.interpolate(label_A.unsqueeze(1).float(), (feat_h, feat_w),
                           mode='nearest').squeeze(1).long()
        lb = F.interpolate(label_B.unsqueeze(1).float(), (feat_h, feat_w),
                           mode='nearest').squeeze(1).long()
        trans_label = (la * self.N + lb).reshape(-1)  # (BHW,)

        p_norm = F.normalize(self.prototypes, dim=-1)
        loss = torch.tensor(0.0, device=q_flat.device)
        count = 0

        for k in range(self.N2):
            mask_k = (trans_label == k)
            if mask_k.sum() < 2:
                continue

            q_region = q_flat[mask_k]
            center = q_region.mean(dim=0)
            center = F.normalize(center, dim=0)

            pos_sim = (center * p_norm[k]).sum()
            pull = 1.0 - pos_sim

            neg_sims = torch.mm(center.unsqueeze(0), p_norm.t()).squeeze(0)
            neg_sims[k] = -1e9
            hardest_neg = neg_sims.max()
            push = F.relu(hardest_neg - pos_sim + self.margin)

            loss = loss + pull + push
            count += 1

        if count > 0:
            loss = loss / count
        return loss


class TransitionAwareComposition(nn.Module):
    """
    Eq.(11): y^scd = h(y1_ss, y2_ss, r, y_tr, m)
    Fuses timestamp-wise semantics, transition features, transition
    prediction, and soft change prior into the final SCD output.
    """
    def __init__(self, num_classes=7, feat_dim=64, query_dim=128):
        super().__init__()
        self.num_transitions = num_classes * num_classes

        in_ch = feat_dim * 2 + query_dim + self.num_transitions + 1
        self.compose = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_transitions, 1),
        )

    def forward(self, dec1, dec2, r_spatial, y_tr, m, out_size):
        r_up = F.interpolate(r_spatial, dec1.shape[2:], mode='bilinear', align_corners=True)
        y_tr_down = F.interpolate(y_tr, dec1.shape[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([dec1, dec2, r_up, y_tr_down, m], dim=1)
        y_scd = self.compose(cat)
        return F.interpolate(y_scd, out_size, mode='bilinear', align_corners=True)