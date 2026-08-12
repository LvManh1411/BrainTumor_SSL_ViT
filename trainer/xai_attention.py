import sys
import os

# Thêm thư mục gốc vào đường dẫn tìm kiếm của Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
import yaml

from models.classifier import BrainTumorClassifier

def generate_attention_map(model, image_tensor, device):
    """
    Trích xuất bản đồ chú ý (Attention Map) từ Vision Transformer (ViT)
    bằng cách tính Cosine Similarity giữa CLS Token và các Patch Tokens.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 1. Trích xuất patch embeddings từ ảnh [batch_size, 196, hidden_dim]
        x = model.encoder._process_input(image_tensor)
        
        # 2. Ghép [CLS] token vào vị trí đầu tiên -> [batch_size, 197, hidden_dim]
        n = x.shape[0]
        batch_class_token = model.encoder.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 3. Đưa qua Transformer Encoder (lúc này đã đủ 197 tokens)
        encoder_out = model.encoder.encoder(x)
        
        # Token index 0 = [CLS] token, các token 1..196 là các Patch 14x14
        cls_token = encoder_out[:, 0:1, :]       # [1, 1, hidden_dim]
        patch_tokens = encoder_out[:, 1:, :]    # [1, 196, hidden_dim]
        
        # Chuẩn hóa L2 vectors
        cls_norm = F.normalize(cls_token, p=2, dim=-1)
        patch_norm = F.normalize(patch_tokens, p=2, dim=-1)
        
        # Tính tương đồng Cosine giữa [CLS] và từng Patch
        similarity = torch.bmm(patch_norm, cls_norm.transpose(1, 2)).squeeze() # [196]
        
        # Reshape từ 1D (196) về ma trận 2D (14x14)
        attn_map = similarity.reshape(14, 14).cpu().numpy()
        
        # Chuẩn hóa ma trận về khoảng [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
    return attn_map

def run_xai():
    print(" BẮT ĐẦU TRỰC QUAN HÓA XAI ATTENTION MAP CHO MÔ HÌNH ViT...\n")
    
    # 1. Đọc file cấu hình
    config_path = "configs/finetune.yaml"
    if not os.path.exists(config_path):
        print(f" Không tìm thấy file config tại: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Thiết bị xử lý: {device}")
    
    # 2. Nạp Mô hình & Trọng số đã huấn luyện
    weights_path = "checkpoints/finetune/best_classifier.pth"
    if not os.path.exists(weights_path):
        print(f" Không tìm thấy trọng số tại: {weights_path}")
        return
        
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    model = BrainTumorClassifier(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f" Đã nạp thành công trọng số: {weights_path}")
    
    # 3. Đọc dữ liệu tập Test
    data_dir = config["dataset"]["data_dir"]
    test_dir = os.path.join(data_dir, "Testing") if os.path.exists(os.path.join(data_dir, "Testing")) else data_dir
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo đồ thị so sánh 4 lớp
    fig, axes = plt.subplots(len(classes), 3, figsize=(11, 3.8 * len(classes)))
    
    for idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_folder):
            print(f" Không tìm thấy thư mục: {cls_folder}")
            continue
            
        img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            continue
            
        # Lấy 1 ảnh mẫu đại diện trong mỗi lớp
        img_path = os.path.join(cls_folder, img_files[0])
        orig_img = Image.open(img_path).convert("RGB")
        orig_img_resized = orig_img.resize((224, 224))
        
        # Tiền xử lý ảnh
        input_tensor = transform(orig_img).unsqueeze(0)
        
        # Dự đoán nhãn
        with torch.no_grad():
            output = model(input_tensor.to(device))
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = classes[pred_idx]
            prob = F.softmax(output, dim=1)[0][pred_idx].item() * 100
        
        # Tạo Attention Map
        attn_map = generate_attention_map(model, input_tensor, device)
        
        # Phóng to Attention Map từ (14x14) lên (224x224) bằng nội suy CUBIC
        attn_map_resized = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Phủ Heatmap lên ảnh MRI gốc
        orig_np = np.array(orig_img_resized)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        
        # Cột 1: Ảnh gốc
        axes[idx, 0].imshow(orig_img_resized)
        axes[idx, 0].set_title(f"Ảnh gốc MRI: {cls_name}", fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Cột 2: Heatmap Chú ý
        axes[idx, 1].imshow(attn_map_resized, cmap='jet')
        axes[idx, 1].set_title("Attention Heatmap", fontsize=11)
        axes[idx, 1].axis('off')
        
        # Cột 3: Overlay Heatmap + Kết quả Dự đoán
        axes[idx, 2].imshow(overlay)
        status_color = 'green' if pred_label == cls_name else 'red'
        axes[idx, 2].set_title(f"Dự đoán: {pred_label} ({prob:.1f}%)", fontsize=11, color=status_color, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.suptitle("Trực quan hóa vùng chú ý (XAI Attention Map) của mô hình ViT trên ảnh MRI U Não", 
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    save_path = "results/xai_attention_maps.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print(f" ĐÃ XUẤT VÀ LƯU ẢNH ATTENTION MAP THÀNH CÔNG TẠI:")
    print(f" {save_path}")
    print("="*60)

if __name__ == "__main__":
    run_xai()