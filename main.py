import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from models import CustomAugmentations, SimSiam
from joint_trainer import JointPretrainer, PseudoLabelDataset
from similarity_learner import SimilarityTrainer
from models import device


# ================= Datasets =================
class SimSiamMNISTDataset(Dataset):
    def __init__(self, base_data, augmentation):
        self.data = base_data
        self.aug = augmentation

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx]).float()
        return self.aug(img), self.aug(img)

    def __len__(self): return len(self.data)


class SiamesePairDataset(Dataset):
    def __init__(self, x0, x1, label):
        self.x0 = torch.from_numpy(x0).float()
        self.x1 = torch.from_numpy(x1).float()
        self.label = torch.from_numpy(label).long()

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx], self.label[idx]

    def __len__(self): return len(self.label)


# ================= Data Loading & Processing (彻底修复) =================
def load_and_process_data(excel_path):
    print("📥 加载数据...")
    fm = pd.read_excel(excel_path)

    # ✅ 修复1：显式提取特征列，彻底避开 drop() 报错与隐藏列干扰
    exclude_cols = ['Label', 'time']
    exclude_cols = [c for c in exclude_cols if c in fm.columns]
    feature_cols = [c for c in fm.columns if c not in exclude_cols]
    print(f"✅ 识别到 {len(feature_cols)} 个特征列，已自动过滤: {exclude_cols}")

    # ✅ 修复2：严格基于特征列计算极值，解决 17 vs 16 维度不匹配
    origin = fm[feature_cols].values.astype(np.float32)
    min_v = torch.tensor(origin.min(axis=0))
    max_v = torch.tensor(origin.max(axis=0))

    def normalize(df):
        arr = df[feature_cols].values.astype(np.float32)
        t = torch.tensor(arr)
        return 2 * (t - min_v) / (max_v - min_v) - 1

    train_df = fm[fm['time'] <= 480]
    test_df = fm[fm['time'] > 480]
    train_x, train_y = normalize(train_df), train_df['Label'].values
    test_x, test_y = normalize(test_df), test_df['Label'].values

    def sliding_window(seq, win_size=16, stride=1):
        return torch.stack([seq[i:i + win_size] for i in range(0, len(seq) - win_size + 1, stride)])

    def build_windows(x, y, stride):
        data, labels = [], []
        num_classes = len(np.unique(y))
        for c in range(num_classes):
            mask = y == c
            xc = x[mask]
            for i in range(0, len(xc), 480):
                chunk = xc[i:i + 480]
                if len(chunk) < 16: continue
                win = sliding_window(chunk)
                data.append(win)
                labels.append(np.full(len(win), c))
        return torch.cat(data), np.concatenate(labels)

    tr_data, tr_label = build_windows(train_x, train_y, stride=1)
    te_data, te_label = build_windows(test_x, test_y, stride=10)

    idx = torch.randperm(len(tr_data))
    tr_data, tr_label = tr_data[idx], tr_label[idx]

    labeled_x, labeled_y, unlabeled_x = [], [], []
    num_classes = len(np.unique(tr_label))
    for c in range(num_classes):
        c_idx = np.where(tr_label == c)[0]
        labeled_x.append(tr_data[c_idx[:5]])
        labeled_y.append(tr_label[c_idx[:5]])
        unlabeled_x.append(tr_data[c_idx[5:]])

    # ✅ 修复3：安全转换维度 (N,16,F) -> (N,1,16,16)，全程 Tensor 操作后转 Numpy
    def to_cnn_format(data_tensor):
        return data_tensor[:, :, :16].unsqueeze(1).float().numpy()

    return (
        to_cnn_format(torch.cat(unlabeled_x)),
        to_cnn_format(torch.cat(labeled_x)),
        np.concatenate(labeled_y),
        to_cnn_format(te_data),
        te_label
    )


def create_pairs(data, labels):
    num_classes = len(np.unique(labels))
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    x0, x1, y = [], [], []
    n = min([len(d) for d in digit_indices]) - 1
    for d in range(num_classes):
        for i in range(n):
            x0.append(data[digit_indices[d][i]])
            x1.append(data[digit_indices[d][i + 1]])
            y.append(1)
            dn = (d + np.random.randint(1, num_classes)) % num_classes
            x0.append(data[digit_indices[d][i]])
            x1.append(data[digit_indices[dn][i]])
            y.append(0)
    return np.array(x0), np.array(x1), np.array(y)


# ================= Main =================
def main():
    print(f"🔧 使用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # 1. 加载数据
    unlabeled_data, labeled_data, labeled_labels, test_data, test_label = load_and_process_data(
        r"C:\Users\HNU\Desktop\zero-shot\data\ahu_similarity_chosefeature.xlsx"
    )

    aug = CustomAugmentations(p=0.8)
    unsup_loader = DataLoader(SimSiamMNISTDataset(unlabeled_data, aug), batch_size=256, shuffle=True)

    x0, x1, y = create_pairs(labeled_data, labeled_labels)
    train_loader = DataLoader(SiamesePairDataset(x0, x1, y), batch_size=5, shuffle=True)

    val_x0, val_x1, val_y = create_pairs(test_data, test_label)
    val_loader = DataLoader(SiamesePairDataset(val_x0, val_x1, val_y), batch_size=5, shuffle=False)

    # 2. 初始化模型
    sim_model = SimSiam().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 3. 阶段一：联合预训练 (对比 + 伪标签)
    print("\n=== > 开始 SimSiam + Pseudo-label 联合预训练 ===")

    # ... (数据加载与模型初始化保持不变) ...

    print("\n=== > 开始无标签数据联合预训练 (对比学习 + 6视图伪标签) ===")
    joint_trainer = JointPretrainer(
        sim_model,
        device=device,
        num_pseudo_clusters=64,  # 伪簇数，仅影响预训练粒度，与15类无关
        lr=5e-2,
        weight=1.0
    )

    pseudo_dataset = PseudoLabelDataset(
        unlabeled_data=unlabeled_data,
        augmentation=aug,
        model=sim_model,
        device=device,
        num_pseudo_clusters=64,
        conf_threshold=0.85  # 置信度阈值，过滤低质量伪标签
    )
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=256, shuffle=True)

    joint_trainer.train_epoch(pseudo_loader, aug=aug, epochs=180)




    # 4. 阶段二：相似性学习与评估
    print("\n=== > 开始 Siamese 相似性学习 ===")
    sim_trainer = SimilarityTrainer(sim_model.encoder, device, lr=0.001)
    sim_trainer.train(train_loader, val_loader, epochs=100)

    print("\n=== > 开始开集诊断评估 ===")
    sim_trainer.evaluate(test_data, test_label, threshold=0.5)

    print("\n🎉 全流程执行完毕！")


if __name__ == "__main__":
    main()