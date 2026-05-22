import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 数据增强器 =================
class CustomAugmentations:
    def __init__(self, p=0.8):
        self.p = p

    def gen_gaussian_noise(self, x, mean=0., std=0.1):
        return x + torch.randn_like(x) * std + mean

    def scaling(self, x, scale_range=(0, 0.1)):
        return x * random.uniform(*scale_range)

    def magnitude_warping(self, x, sigma=0.2, knot=4):
        W = x.shape[-1]
        warp = torch.tensor(np.interp(np.arange(W),
                                      np.linspace(0, W, knot),
                                      np.random.normal(loc=1.0, scale=sigma, size=knot))).float().to(x.device)
        # ✅ 修复：改为 3D 视图 (1, 1, -1)，避免广播补出第2维
        return x * warp.view(1, 1, -1)

    def zero_making_by_feature(self, x, p=0.2):
        return x * (torch.rand_like(x) > p).float()

    def permutation(self, x, n_segments=4):
        indices = list(range(n_segments))
        random.shuffle(indices)
        chunks = torch.chunk(x, n_segments, dim=-1)
        return torch.cat([chunks[i] for i in indices], dim=-1)

    def __call__(self, x):
        # ✅ 修复：入口强制标准化为 (C, H, W) 即 (1, 16, 16)
        if x.ndim == 4:
            x = x.squeeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0)

        if random.random() < self.p: x = self.gen_gaussian_noise(x)
        if random.random() < self.p: x = self.scaling(x)
        if random.random() < self.p: x = self.magnitude_warping(x)
        if random.random() < self.p: x = self.zero_making_by_feature(x)
        if random.random() < self.p: x = self.permutation(x)
        return torch.clamp(x, 0., 1.)  # 稳定返回 (1, 16, 16)


# ================= 网络结构 =================
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x): return self.net(x)


class SimpleConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


class SimSiam(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = SimpleConvEncoder()
        self.projector = MLPHead(256, 512, feature_dim)
        self.predictor = MLPHead(feature_dim, 256, feature_dim)

    def forward(self, x1, x2):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, z2.detach(), p2, z1.detach()


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 10), nn.Linear(10, 2)
        )

    def forward_once(self, x): return self.fc1(self.encoder(x))

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)


# ================= 损失函数 =================
def simsiam_loss(p1, z2, p2, z1):
    def D(p, z):
        p, z = F.normalize(p, dim=1), F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    return D(p1, z2) / 2 + D(p2, z1) / 2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        mdist = self.margin - torch.sqrt(dist_sq)
        dist = torch.clamp(mdist, min=0.0)
        return (y * dist_sq + (1 - y) * torch.pow(dist, 2)).mean()