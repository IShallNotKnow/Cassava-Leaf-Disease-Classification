import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = in_channels // reduction
        deep = mid // 2

        # use nn.Linear only — no nn.ReLU modules
        self.fc1 = nn.Linear(in_channels, mid)
        self.fc2 = nn.Linear(mid, deep)
        self.fc3 = nn.Linear(deep, in_channels)

    def forward(self, x):
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))

        # F.relu instead of nn.ReLU — invisible to guided backprop hook registration
        def mlp(t):
            return self.fc3(F.relu(self.fc2(F.relu(self.fc1(t)))))

        weight = torch.sigmoid(mlp(avg) + mlp(mx))
        return x * weight.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class GeM(nn.Module):
    """Generalized Mean Pooling — learns to emphasize discriminative regions
    more than global average pooling which weights everything equally"""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            1
        ).pow(1.0 / self.p)


class ModifiedCassavaNet(nn.Module):
    def __init__(self, num_classes=5, proj_dim=128):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # EfficientNetB0 feature blocks:
        # features[0] = stem conv
        # features[1-6] = MBConv blocks
        # features[7] = head conv
        # Attach aux head at block 5 output — 2/3 depth, 80 channels
        self.early_features = nn.Sequential(*list(self.backbone.features[:6]))
        self.late_features = nn.Sequential(*list(self.backbone.features[6:]))

        # Aux head at block 5 — 112 channels at this point
        aux_in = 112  # MBConv block 5 output channels for B0
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(aux_in),
            nn.Dropout(0.3),
            nn.Linear(aux_in, num_classes)
        )

        self.dropouts = nn.ModuleList(
            [nn.Dropout(0.2 + 0.05 * i) for i in range(5)]
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, num_classes)
        )

        # Projection head — only used during training for SupCon
        # maps 1280 → 256 → 128, L2 normalized
        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

        self.cbam = CBAM(in_channels=1280, reduction=16)
        self.gem_pool = GeM()

    def forward(self, x):
        x = self.early_features(x)
        aux_out = self.aux_head(x)  # supervises mid-level features

        x = self.late_features(x)
        x = self.cbam(x)
        x = self.gem_pool(x)
        x = torch.flatten(x, 1)
        features = x

        # BN first, then dropout, then linear
        x_bn = self.head[0](x)
        out = sum(
            self.head[1](drop(x_bn))
            for drop in self.dropouts
        ) / len(self.dropouts)

        proj = F.normalize(self.projector(features), dim=1)

        return out, aux_out, proj

    def extract_features(self, x):
        x = self.early_features(x)
        x = self.late_features(x)
        x = self.cbam(x)
        x = self.gem_pool(x)
        x = torch.flatten(x, 1)
        return x

    def guided_forward(self, x):
        """Forward pass without CBAM for guided backprop compatibility"""
        x = self.early_features(x)
        aux_out = self.aux_head(x)

        x = self.late_features(x)
        # skip cbam — its MLP causes shape mismatch in guided backprop backward
        x = self.gem_pool(x)
        x = torch.flatten(x, 1)

        x_bn = self.head[0](x)
        out = sum(
            self.head[1](drop(x_bn))
            for drop in self.dropouts
        ) / len(self.dropouts)

        return out, aux_out, None
