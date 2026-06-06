import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import build_resnet34
from models.layers import (
    Multi_Level_Feature_Aggregation, CBA1x1, CBA3x3,
    SoftChangePrior, TransitionQueryConstructor,
    TransitionPrototypeReasoning, TransitionAwareComposition,
    SemanticDecoder,
)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        resnet = build_resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.mlfa = Multi_Level_Feature_Aggregation()

    def forward(self, x):
        x = self.layer0(x)       # H/2
        x = self.maxpool(x)      # H/4
        x_low = self.layer1(x)   # H/4, 64ch
        x1 = self.layer2(x_low)  # H/8, 128ch
        x2 = self.layer3(x1)     # H/8, 256ch (dilated)
        x3 = self.layer4(x2)     # H/8, 512ch (dilated)
        x = self.mlfa(x1, x2, x3)  # H/8, 128ch
        return x, x_low


class TransSCD(nn.Module):
    def __init__(self, in_channels=3, num_classes=7,
                 feat_dim=128, query_dim=128, tau=0.07,
                 prior_lambda=1.0, margin=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_transitions = num_classes * num_classes

        self.encoder = FCN(in_channels, pretrained=True)

        self.sem_decoder = SemanticDecoder(feat_dim, 64)
        self.sem_head = nn.Conv2d(64, num_classes, 1, bias=False)

        self.prior_branch = SoftChangePrior(feat_dim)

        self.query_constructor = TransitionQueryConstructor(feat_dim, query_dim)

        self.prototype_reasoning = TransitionPrototypeReasoning(
            num_classes=num_classes,
            query_dim=query_dim,
            tau=tau,
            prior_lambda=prior_lambda,
            margin=margin,
        )

        self.transition_head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, self.num_transitions),
        )

        self.composition = TransitionAwareComposition(
            num_classes=num_classes,
            feat_dim=64,
            query_dim=query_dim,
        )

    def forward(self, x1, x2, label_A=None, label_B=None):
        x_size = x1.size()

        s1, s1_low = self.encoder(x1)
        s2, s2_low = self.encoder(x2)

        dec1 = self.sem_decoder(s1, s1_low)
        dec2 = self.sem_decoder(s2, s2_low)

        out1 = F.interpolate(self.sem_head(dec1), x_size[2:], mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.sem_head(dec2), x_size[2:], mode='bilinear', align_corners=True)

        m = self.prior_branch(s1, s2)

        q = self.query_constructor(s1, s2)

        B, C, H, W = q.shape
        q_flat = q.permute(0, 2, 3, 1).reshape(B * H * W, C)
        m_flat = m.squeeze(1).reshape(B * H * W)

        r, consistency_loss = self.prototype_reasoning(
            q_flat, m_flat, B, H, W,
            label_A=label_A, label_B=label_B,
        )

        y_tr = self.transition_head(r)
        y_tr = y_tr.reshape(B, H, W, self.num_transitions).permute(0, 3, 1, 2)
        y_tr = F.interpolate(y_tr, x_size[2:], mode='bilinear', align_corners=True)

        r_spatial = r.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        m_up = F.interpolate(m, dec1.shape[2:], mode='bilinear', align_corners=True)

        y_scd = self.composition(dec1, dec2, r_spatial, y_tr, m_up, x_size[2:])

        m_out = F.interpolate(m, x_size[2:], mode='bilinear', align_corners=True).squeeze(1)

        return out1, out2, m_out, y_tr, y_scd, consistency_loss


if __name__ == '__main__':
    x1 = torch.randn(2, 3, 512, 512).cuda().float()
    x2 = torch.randn(2, 3, 512, 512).cuda().float()

    model = TransSCD(3, num_classes=7).cuda()
    model.eval()

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, (x1, x2))
    total = sum(p.nelement() for p in model.parameters())
    print("Params: %.2fM" % (total / 1e6))
    print("FLOPs: %.2fG" % (flops.total() / 1e9))

    with torch.no_grad():
        for _ in range(10):
            _ = model(x1, x2)

    start = time.time()
    with torch.no_grad():
        output = model(x1, x2)
    print(f"Inference time: {(time.time() - start) * 1000:.2f} ms")
