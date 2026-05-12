import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class PseudoLabelTrainer:
    """
    伪标签预测模块：在预训练阶段，对无标签数据生成高置信度伪标签，
    与对比学习损失联合优化，提升特征判别性。
    """

    def __init__(self, model, device, num_classes=15, conf_threshold=0.95, weight=0.5):
        """
        Args:
            model: 共享编码器 + 临时分类头
            device: 计算设备
            num_classes: 已知故障类别数
            conf_threshold: 伪标签置信度阈值（仅保留高置信样本）
            weight: 伪标签损失权重（与对比学习损失权重相等，默认 0.5）
        """
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.weight = weight  # λ_pseudo = 0.5

        # 临时分类头（仅用于伪标签生成，不参与最终推理）
        self.temp_classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, num_classes)
        ).to(device)

        # 优化器：同时更新 encoder + temp_classifier
        self.optimizer = torch.optim.AdamW(
            list(model.encoder.parameters()) + list(self.temp_classifier.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _generate_pseudo_labels(self, x):
        """生成伪标签：仅保留高置信度预测"""
        with torch.no_grad():
            feat = self.model.encoder(x).view(x.size(0), -1)
            logits = self.temp_classifier(feat)
            probs = F.softmax(logits, dim=1)
            max_conf, pseudo = torch.max(probs, dim=1)
            # 掩码：仅保留置信度 > threshold 的样本
            mask = max_conf >= self.conf_threshold
        return pseudo, max_conf, mask

    def train_step(self, unlabeled_loader, contrastive_loss_fn, contrastive_weight=0.5):
        """
        单步联合训练：对比损失 + 伪标签损失
        Args:
            unlabeled_loader: 无标签数据 DataLoader（返回 (x1, x2) 正样本对）
            contrastive_loss_fn: SimSiam 损失函数
            contrastive_weight: 对比学习损失权重（默认 0.5，与伪标签权重相等）
        """
        self.model.train()
        self.temp_classifier.train()

        total_loss = 0.0
        total_contrast = 0.0
        total_pseudo = 0.0
        pseudo_count = 0

        for x1, x2 in unlabeled_loader:
            x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
            batch_size = x1.size(0)

            # ========== 1. 对比学习损失 (SimSiam) ==========
            p1, z2, p2, z1 = self.model(x1, x2)
            loss_contrast = contrastive_loss_fn(p1, z2, p2, z1)

            # ========== 2. 伪标签预测损失 ==========
            # 对 x1 生成伪标签
            pseudo, conf, mask = self._generate_pseudo_labels(x1)

            if mask.sum() > 0:
                # 仅对高置信样本计算伪标签损失
                feat = self.model.encoder(x1[mask]).view(-1, 1024)
                logits = self.temp_classifier(feat)
                loss_pseudo = self.criterion(logits, pseudo[mask])
                # 置信度加权：越确定的样本，损失权重越大
                loss_pseudo = (loss_pseudo * conf[mask]).mean()
            else:
                loss_pseudo = torch.tensor(0.0, device=self.device)

            # ========== 3. 联合损失：权重相等 ==========
            total_loss_batch = contrastive_weight * loss_contrast + self.weight * loss_pseudo

            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            # 日志统计
            total_loss += total_loss_batch.item()
            total_contrast += loss_contrast.item()
            total_pseudo += loss_pseudo.item() if mask.sum() > 0 else 0.0
            pseudo_count += mask.sum().item()

        return {
            'loss': total_loss / len(unlabeled_loader),
            'contrast_loss': total_contrast / len(unlabeled_loader),
            'pseudo_loss': total_pseudo / len(unlabeled_loader),
            'pseudo_ratio': pseudo_count / (
                len(unlabeled_loader.dataset) if hasattr(unlabeled_loader.dataset, '__len__') else 1)
        }

    def update_classifier_head(self, labeled_loader, epochs=5):
        """
        用少量有标签数据微调临时分类头，提升伪标签质量（可选）
        """
        print("🔄 Updating pseudo-label classifier with labeled data...")
        self.temp_classifier.train()
        optimizer_clf = torch.optim.Adam(self.temp_classifier.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for x, y in labeled_loader:
                x, y = x.float().to(self.device), y.long().to(self.device)
                feat = self.model.encoder(x).view(x.size(0), -1)
                logits = self.temp_classifier(feat)
                loss = F.cross_entropy(logits, y)
                optimizer_clf.zero_grad()
                loss.backward()
                optimizer_clf.step()