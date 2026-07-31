import torch
import torch.nn as nn
from models.vit_model import BrainTumorViT

class SSLViT(nn.Module):
    def __init__(self, pretrained=True, projection_dim=128):
        super(SSLViT, self).__init__()
        
        # 1. Khởi tạo Backbone ViT
        base_model = BrainTumorViT(num_classes=4, pretrained=pretrained)
        
        # Lấy phần trích xuất đặc trưng của ViT (bỏ lớp classifier head 4 classes cuối)
        self.encoder = base_model.vit
        in_features = self.encoder.heads.head.in_features
        self.encoder.heads.head = nn.Identity() # Loại bỏ head cũ
        
        # 2. Xây dựng Projection Head (MLP) phục vụ SSL (SimCLR / Barlow Twins style)
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        # Trích xuất đặc trưng ẩn từ ViT Encoder
        features = self.encoder(x)
        # Bắn qua Projection Head để tính SSL Loss
        projections = self.projection_head(features)
        return features, projections


# --- ĐOẠN CODE TEST NHANH MODULE SSL ---
if __name__ == "__main__":
    # Khởi tạo mô hình SSL ViT
    ssl_model = SSLViT(pretrained=False, projection_dim=128)
    
    # Giả lập 1 batch gồm 16 ảnh MRI đầu vào
    dummy_input = torch.randn(16, 3, 224, 224)
    
    # Chạy qua mô hình
    features, projections = ssl_model(dummy_input)
    
    print("Khởi tạo module SSL ViT thành công!")
    print(f"Kích thước Feature Vector từ ViT Encoder: {features.shape}")      # [16, 768]
    print(f"Kích thước Projection Vector phục vụ SSL Loss: {projections.shape}") # [16, 128]