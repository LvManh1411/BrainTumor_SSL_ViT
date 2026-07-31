import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class BrainTumorViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(BrainTumorViT, self).__init__()
        
        # 1. Nạp trọng số tiền huấn luyện (Pretrained Weights) của ViT-B/16
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        
        # 2. Thay thế lớp Classifier Head cuối cùng cho bài toán 4 class
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


# --- TEST NHANH KẾT NỐI MÔ HÌNH ---
if __name__ == "__main__":
    # Khởi tạo mô hình (tắt pretrained để test cho nhanh)
    model = BrainTumorViT(num_classes=4, pretrained=False)
    
    # Tạo 1 batch ảnh giả lập có kích thước [32, 3, 224, 224]
    dummy_input = torch.randn(32, 3, 224, 224)
    
    # Đưa qua mô hình
    output = model(dummy_input)
    
    print("Khởi tạo mô hình Vision Transformer thành công!")
    print(f"Kích thước đầu vào (Batch size, Channels, H, W): {dummy_input.shape}")
    print(f"Kích thước đầu ra (Logits 4 lớp u): {output.shape}") # Mong đợi: [32, 4]