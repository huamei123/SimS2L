import torch
from models import simsiam_loss

class ContrastiveTrainer:
    def __init__(self, model, device, lr=0.05, momentum=0.9):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def train(self, dataloader, epochs=10):
        print(" === 开始 SimSiam 对比学习预训练 ===")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x1, x2 in dataloader:
                x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
                p1, z2, p2, z1 = self.model(x1, x2)
                loss = simsiam_loss(p1, z2, p2, z1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | SimSiam Loss = {total_loss/len(dataloader):.4f}")
        print("✅ 对比学习预训练完成\n")
        return self.model