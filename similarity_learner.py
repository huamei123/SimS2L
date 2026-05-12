import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from models import SiameseNetwork, ContrastiveLoss


class SimilarityTrainer:
    def __init__(self, encoder, device, lr=0.001, num_classes=15):
        self.device = device
        self.classifier = SiameseNetwork(encoder).to(device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = ContrastiveLoss()
        self.num_classes = num_classes


    def train(self, train_loader, val_loader, epochs=200):
        print("🎯 === 开始 Siamese 相似性学习 ===")
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.classifier.train()
            t_loss = 0
            for x1, x2, label in train_loader:
                # ✅ 关键修复：添加 .float() 和 .long() 确保类型一致
                x1, x2, label = x1.float().to(self.device), x2.float().to(self.device), label.long().to(self.device)
                out1, out2 = self.classifier(x1, x2)
                loss = self.criterion(out1, out2, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                t_loss += loss.item()
            train_losses.append(t_loss / len(train_loader))

            self.classifier.eval()
            v_loss = 0
            with torch.no_grad():
                for vx1, vx2, vlabel in val_loader:
                    # ✅ 验证集同样修复
                    vx1, vx2 = vx1.float().to(self.device), vx2.float().to(self.device)
                    out1, out2 = self.classifier(vx1, vx2)
                    v_loss += self.criterion(out1, out2, vlabel.long().to(self.device)).item()
            val_losses.append(v_loss / len(val_loader))

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

        self._plot_loss(train_losses, val_losses)
        print("✅ 相似性学习完成\n")
        return self.classifier

    def evaluate(self, test_data, test_label, threshold=0.175):
        print(" === 开始开集诊断评估 ===")
        self.classifier.eval()
        test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
        test_loader = DataLoader(test_dataset, batch_size=64)

        known_classes = list(range(15))
        class_features = defaultdict(list)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.device)  # ✅ 修复
                feats = self.classifier.forward_once(x).cpu()
                for f, label in zip(feats, y):
                    class_features[label.item()].append(f)
        centers = {k: torch.stack(v).mean(dim=0) for k, v in class_features.items()}

        known_correct, unknown_correct, known_total, unknown_total = 0, 0, 0, 0
        for x, y in test_dataset:
            x = x.float().reshape(-1, 1, 16, 16).to(self.device)  # ✅ 修复
            feat = self.classifier.forward_once(x).cpu()
            dists = {k: F.pairwise_distance(feat, v.unsqueeze(0)).item() for k, v in centers.items()}
            pred_class = min(dists, key=dists.get)
            pred = pred_class if dists[pred_class] <= threshold else "Unknown"

            if y in known_classes:
                known_total += 1
                if pred == y: known_correct += 1
            else:
                unknown_total += 1
                if pred == "Unknown": unknown_correct += 1

        kfa = known_correct / known_total if known_total > 0 else 0
        ufa = unknown_correct / unknown_total if unknown_total > 0 else 0
        print(f" KFA (Known Fault Accuracy): {kfa:.4f}")
        print(f"🛡️ UFA (Unknown Fault Accuracy): {ufa:.4f}")
        return kfa, ufa

    def _plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch');
        plt.ylabel('Loss')
        plt.title('Siamese Training & Validation Loss')
        plt.legend();
        plt.grid(True)
        plt.savefig("siam_loss_curve.png", dpi=300)
        pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Train": train_losses, "Val": val_losses}).to_excel(
            "loss_curve.xlsx", index=False)
        print("📈 损失曲线已保存至 loss_curve.png / .xlsx")