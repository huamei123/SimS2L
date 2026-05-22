import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import simsiam_loss
from sklearn.cluster import KMeans
import numpy as np


class PseudoLabelDataset(Dataset):
    """
    纯无监督伪标签数据集：基于特征聚类生成伪标签，完全独立于下游15类故障
    """

    def __init__(self, unlabeled_data, augmentation, model, device,
                 num_pseudo_clusters=64, conf_threshold=0.85):
        self.data = unlabeled_data
        self.aug = augmentation
        self.model = model
        self.device = device
        self.num_pseudo_clusters = num_pseudo_clusters  # 伪簇数，与下游类别数正交
        self.conf_threshold = conf_threshold
        self.pseudo_labels = None
        self._generate_pseudo_labels()

    def _generate_pseudo_labels(self):
        self.model.eval()
        feats = []
        with torch.no_grad():
            for i in range(len(self.data)):
                x = torch.tensor(self.data[i]).float().to(self.device)
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                elif x.ndim == 4:
                    x = x.squeeze(0)
                # 提取原始视图特征用于聚类
                feat = self.model.encoder(x).view(1, -1).cpu().numpy()
                feats.append(feat)

        feats_np = np.concatenate(feats, axis=0)

        # 无监督聚类生成伪标签 (不依赖任何真实类别)
        kmeans = KMeans(n_clusters=self.num_pseudo_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feats_np)

        # 基于样本到簇中心距离计算置信度
        dists = kmeans.transform(feats_np)
        min_dists = np.min(dists, axis=1)
        max_dist = np.max(min_dists) if np.max(min_dists) > 0 else 1.0
        confidences = 1.0 - (min_dists / max_dist)

        # 高置信样本分配伪标签，低置信标记为 -1 (CrossEntropyLoss 自动忽略)
        self.pseudo_labels = np.where(confidences >= self.conf_threshold, cluster_labels, -1)
        self.pseudo_labels = torch.tensor(self.pseudo_labels, dtype=torch.long)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).float()
        if x.ndim == 2: x = x.unsqueeze(0)
        return x, self.pseudo_labels[idx]

    def __len__(self):
        return len(self.data)


class JointPretrainer:
    """
    联合预训练器：SimSiam对比损失 + 6视图伪标签一致性损失（直接相加）
    仅用于无标签数据预训练，不涉及下游故障分类
    """

    def __init__(self, model, device, num_pseudo_clusters=64, lr=1e-3, weight=1.0):
        self.model = model.to(device)
        self.device = device
        self.num_pseudo_clusters = num_pseudo_clusters

        # 临时聚类分类头：输出维度 = 伪簇数，与下游15类完全解耦
        if not hasattr(model, 'temp_cls'):
            model.temp_cls = torch.nn.Linear(256, self.num_pseudo_clusters).to(device)

        params = list(model.encoder.parameters()) + list(model.temp_cls.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        self.criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.weight = weight  # λ_pseudo = λ_contrast = 1.0

    def _get_6_views(self, x, aug):
        """确定性生成6个视图: 1原始 + 5种增强 (B, 1, 16, 16) -> List[(B, 1, 16, 16)]"""
        if x.ndim == 3: x = x.unsqueeze(1)
        views = [x]
        views.append(aug.gen_gaussian_noise(x))
        views.append(aug.scaling(x))
        views.append(aug.magnitude_warping(x))
        views.append(aug.zero_making_by_feature(x))
        views.append(aug.permutation(x))
        return views

    def train_epoch(self, dataloader, aug, epochs=20):
        self.model.train()
        for epoch in range(epochs):
            total_loss, total_cont, total_pseudo = 0, 0, 0
            valid_cnt = 0

            for x, pseudo_labels in dataloader:
                x = x.float().to(self.device)
                pseudo_labels = pseudo_labels.long().to(self.device)

                # === 1. 动态生成6个视图 ===
                views = self._get_6_views(x, aug)

                # === 2. 6视图伪标签一致性损失 ===
                loss_pseudo = torch.tensor(0.0, device=self.device)
                for v in views:
                    feat = self.model.encoder(v)
                    logits = self.model.temp_cls(feat)
                    loss_pseudo += self.criterion_cls(logits, pseudo_labels)
                loss_pseudo /= len(views)  # 6视图平均

                # === 3. 随机取2个视图计算 SimSiam 对比损失 ===
                idx1, idx2 = torch.randint(0, len(views), (2,))
                v1, v2 = views[idx1], views[idx2]
                p1, z2, p2, z1 = self.model(v1, v2)
                loss_cont = simsiam_loss(p1, z2, p2, z1)

                # === 4. 联合损失：直接相加，权重相等 ===
                loss = loss_cont + self.weight * loss_pseudo

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_cont += loss_cont.item()
                total_pseudo += loss_pseudo.item()
                valid_cnt += 1

            print(f"Joint Pretrain Epoch {epoch + 1}: Loss={total_loss / valid_cnt:.4f} "
                  f"(Cont={total_cont / valid_cnt:.4f}, Pseudo={total_pseudo / valid_cnt:.4f})")