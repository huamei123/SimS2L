import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from models import SiameseNetwork, ContrastiveLoss


class SimilarityTrainer:
    def __init__(self, encoder, device, lr=0.0005):
        self.device = device
        self.classifier = SiameseNetwork(encoder).to(device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = ContrastiveLoss()

    def train(self, train_loader, val_loader, epochs=100):
        print("🎯 === 开始 Siamese 相似性学习 ===")
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # === 训练阶段 ===
            self.classifier.train()
            t_loss = 0
            for x1, x2, label in train_loader:
                # ✅ 确保数据与标签全部在同一设备
                x1 = x1.float().to(self.device)
                x2 = x2.float().to(self.device)
                label = label.long().to(self.device)
                if x1.ndim == 3: x1 = x1.unsqueeze(1)
                if x2.ndim == 3: x2 = x2.unsqueeze(1)

                out1, out2 = self.classifier(x1, x2)
                loss = self.criterion(out1, out2, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                t_loss += loss.item()
            train_losses.append(t_loss / len(train_loader))

            # === 验证阶段 ===
            self.classifier.eval()
            v_loss = 0
            with torch.no_grad():
                for vx1, vx2, vlabel in val_loader:
                    # 🔑 关键修复：vlabel 必须显式移动到 GPU，否则与 out1/out2 设备冲突
                    vx1 = vx1.float().to(self.device)
                    vx2 = vx2.float().to(self.device)
                    vlabel = vlabel.long().to(self.device)

                    if vx1.ndim == 3: vx1 = vx1.unsqueeze(1)
                    if vx2.ndim == 3: vx2 = vx2.unsqueeze(1)

                    out1, out2 = self.classifier(vx1, vx2)
                    v_loss += self.criterion(out1, out2, vlabel).item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train: {train_losses[-1]:.4f}")

        print("✅ 相似性学习完成\n")
        return self.classifier

    def evaluate(self, test_data, test_label, threshold=0.5):
        print(" === 开始开集诊断评估 ===")
        self.classifier.eval()
        test_ds = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
        loader = DataLoader(test_ds, batch_size=64)

        centers = defaultdict(list)
        with torch.no_grad():
            for x, y in loader:
                x = x.float().to(self.device)
                if x.ndim == 3: x = x.unsqueeze(1)
                feat = self.classifier.forward_once(x).cpu()
                for f, l in zip(feat, y): centers[l.item()].append(f)
        centers = {k: torch.stack(v).mean(0) for k, v in centers.items()}
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        k_corr, u_corr, k_tot, u_tot = 0, 0, 0, 0
        for x, y in test_ds:
            x = x.float().to(self.device)
            if x.ndim == 3: x = x.unsqueeze(1)
            feat = self.classifier.forward_once(x).cpu()
            dists = {k: F.pairwise_distance(feat, v.unsqueeze(0)).item() for k, v in centers.items()}
            pred = min(dists, key=dists.get)
            if y in labels:
                k_tot += 1
                if dists[pred] <= threshold and pred == y: k_corr += 1
            else:
                u_tot += 1
                if dists[pred] > threshold: u_corr += 1

        print(f"EVAL -> KFA: {k_corr / k_tot:.4f}, UFA: {u_corr / u_tot:.4f}")
        return k_corr / k_tot, u_corr / u_tot
    #
    # def _plot_loss(self, t, v):
    #     pd.DataFrame({"Train": t, "Val": v}).to_excel("loss_curve.xlsx")
    #     plt.plot(t, label='Train');
    #     plt.plot(v, label='Val')
    #     plt.legend();
    #     plt.savefig("loss_curve.png");
    #     print("Loss saved.")