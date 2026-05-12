import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from models import CustomAugmentations
from contrastive_learner import ContrastiveTrainer
from similarity_learner import SimilarityTrainer
from pseudo_label_prediction import PseudoLabelTrainer


# -------------------------
# 数据加载与预处理
# -------------------------
class SimSiamMNISTDataset(Dataset):
    def __init__(self, base_data, augmentation):
        self.data = base_data
        self.aug = augmentation

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx])
        return self.aug(img), self.aug(img)

    def __len__(self): return len(self.data)


class SiamesePairDataset(Dataset):
    def __init__(self, x0, x1, label):
        self.x0, self.x1, self.label = torch.from_numpy(x0), torch.from_numpy(x1), torch.from_numpy(label)

    def __getitem__(self, idx): return self.x0[idx], self.x1[idx], self.label[idx]

    def __len__(self): return len(self.label)


def load_and_process_data(excel_path):
    print("📥 加载数据...")
    fm = pd.read_excel(excel_path)
    train_df = fm[fm['time'] <= 480].drop(columns=['time'])
    test_df = fm[fm['time'] > 480].drop(columns=['time'])

    def normalize(df, min_v, max_v):
        arr = df.drop(columns=['Label']).values
        t = torch.tensor(arr)
        return 2 * (t - min_v) / (max_v - min_v) - 1

    origin = torch.tensor(fm.drop(columns=['Label', 'time']).values)
    min_v, max_v = origin.min(dim=0).values, origin.max(dim=0).values
    train_x = normalize(train_df, min_v, max_v)
    test_x = normalize(test_df, min_v, max_v)
    train_y = train_df['Label'].values
    test_y = test_df['Label'].values

    def sliding_window(seq, win_size=16, stride=1):
        return torch.stack([seq[i:i + win_size] for i in range(0, len(seq) - win_size + 1, stride)])

    def build_windows(x, y, stride):
        data, labels = [], []
        for c in range(16):
            mask = y == c
            x_c, y_c = x[mask], y[mask]
            for i in range(0, len(x_c), 480):
                chunk = x_c[i:i + 480]
                if len(chunk) < 16: continue
                win = sliding_window(chunk)
                data.append(win)
                labels.append(np.full(len(win), c))
        return torch.cat(data), np.concatenate(labels)

    tr_data, tr_label = build_windows(train_x, train_y, stride=1)
    te_data, te_label = build_windows(test_x, test_y, stride=10)

    # 划分已知/未知 & 采样少量标签
    idx = torch.randperm(len(tr_data))
    tr_data, tr_label = tr_data[idx], tr_label[idx]
    labeled_x, labeled_y, unlabeled_x, unlabeled_y = [], [], [], []

    for c in range(15):
        c_idx = np.where(tr_label == c)[0]
        labeled_x.append(tr_data[c_idx[:5]])
        labeled_y.append(tr_label[c_idx[:5]])
        unlabeled_x.append(tr_data[c_idx[5:]])
        unlabeled_y.append(tr_label[c_idx[5:]])

    labeled_data = torch.cat(labeled_x).numpy()
    labeled_labels = np.concatenate(labeled_y)
    unlabeled_data = torch.cat(unlabeled_x).numpy()

    return unlabeled_data, labeled_data, labeled_labels, te_data.numpy(), te_label


def create_pairs(data, labels, batchsize=32):
    # 简化版配对生成，适配 Siamese 训练
    digit_indices = [np.where(labels == i)[0] for i in range(15)]
    x0, x1, y = [], [], []
    n = min([len(d) for d in digit_indices]) - 1
    for d in range(15):
        for i in range(n):
            x0.append(data[digit_indices[d][i]])
            x1.append(data[digit_indices[d][i + 1]])
            y.append(1)
            dn = (d + np.random.randint(1, 15)) % 15
            x0.append(data[digit_indices[d][i]])
            x1.append(data[digit_indices[dn][i]])
            y.append(0)
    return np.array(x0), np.array(x1), np.array(y)


# -------------------------
# 主流程
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"️ 使用设备: {device}")

    # 1. 数据准备
    unlabeled_data, labeled_data, labeled_labels, test_data, test_label = load_and_process_data(
        r"C:\Users\HNU\Desktop\zero-shot\data\ahu_similarity_chosefeature.xlsx"
    )
    aug = CustomAugmentations(p=0.8)
    unsup_loader = DataLoader(SimSiamMNISTDataset(unlabeled_data, aug), batch_size=256, shuffle=True)

    # 构造配对数据
    x0, x1, y = create_pairs(labeled_data, labeled_labels)
    train_set = SiamesePairDataset(x0, x1, y)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    val_x0, val_x1, val_y = create_pairs(test_data, test_label)
    val_set = SiamesePairDataset(val_x0, val_x1, val_y)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 2. 初始化模型
    from models import SimSiam
    sim_model = SimSiam().to(device)

    # 3. 阶段一：对比学习
    contrast_trainer = ContrastiveTrainer(sim_model, device, lr=0.05)
    pretrained_model = contrast_trainer.train(unsup_loader, epochs=10)

    # 4. 阶段二：相似性学习 + 评估
    sim_trainer = SimilarityTrainer(pretrained_model.encoder, device, lr=0.001)
    sim_trainer.train(train_loader, val_loader, epochs=200)
    sim_trainer.evaluate(test_data, test_label, threshold=0.175)

    print("\n🎉 全流程执行完毕！")


if __name__ == "__main__":
    main()