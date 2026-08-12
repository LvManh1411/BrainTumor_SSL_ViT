import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Định nghĩa NT-Xent Loss (SimCLR Loss) cho SSL
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize vector đặc trưng
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Tính Cosine Similarity giữa các cặp ảnh
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Tạo mask loại bỏ điểm tự so sánh với chính nó
        batch_size = z_i.shape[0]
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity_matrix / self.temperature).masked_fill(mask, 0).sum(dim=1)
        
        loss = -torch.log(nominator / denominator).mean()
        return loss

# 2. Lớp Trainer quản lý vòng lặp huấn luyện
class SSLTrainer:
    def __init__(self, model, train_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.criterion = NTXentLoss(temperature=config["train"]["temperature"])
        self.save_dir = config["train"]["save_dir"]
        
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # Tạo 2 góc nhìn (views) khác nhau của cùng 1 batch ảnh để học tương phản
            x_i = images 
            x_j = torch.flip(images, dims=[3]) # Lật ngang làm view thứ 2
            
            self.optimizer.zero_grad()
            
            _, z_i = self.model(x_i)
            _, z_j = self.model(x_j)
            
            loss = self.criterion(z_i, z_j)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f" Epoch [{epoch+1}/{self.config['train']['epochs']}] | Batch [{batch_idx+1}/{len(self.train_loader)}] | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def fit(self):
        print("\n BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN SSL...")
        for epoch in range(self.config["train"]["epochs"]):
            avg_loss = self.train_epoch(epoch)
            print(f"=== Kết thúc Epoch {epoch+1} | Loss trung bình: {avg_loss:.4f} ===")
            
            # Lưu Checkpoint trọng số mô hình
            checkpoint_path = os.path.join(self.save_dir, f"ssl_vit_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f" Đã lưu Checkpoint tại: {checkpoint_path}\n")