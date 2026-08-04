import torch
import torch.nn as nn
from models.vit_model import BrainTumorViT

class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, ssl_checkpoint_path=None, freeze_backbone=False):
        super(BrainTumorClassifier, self).__init__()
        
        # 1. Khởi tạo Backbone ViT gốc
        base_model = BrainTumorViT(num_classes=num_classes, pretrained=False)
        self.encoder = base_model.vit
        
        # 2. Nạp trọng số từ SSL Pretraining
        if ssl_checkpoint_path:
            self.load_ssl_weights(ssl_checkpoint_path)
            
        # 3. Đóng băng backbone nếu chỉ muốn train classification head
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 4. Thay thế Head phân loại cuối cùng cho 4 lớp
        in_features = self.encoder.heads.head.in_features
        self.encoder.heads.head = nn.Linear(in_features, num_classes)

    def load_ssl_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder_state_dict = {}
        
        # Lọc trích xuất các weight thuộc về encoder từ SSL model
        for k, v in checkpoint.items():
            if k.startswith("encoder."):
                encoder_state_dict[k.replace("encoder.", "")] = v
                
        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f" Đã nạp thành công trọng số SSL từ: {checkpoint_path}")

    def forward(self, x):
        return self.encoder(x)


# --- TEST NHANH MODULE CLASSIFIER ---
if __name__ == "__main__":
    model = BrainTumorClassifier(num_classes=4)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f" Khởi tạo Classifier thành công! Đầu ra Output shape: {output.shape}") # [4, 4]