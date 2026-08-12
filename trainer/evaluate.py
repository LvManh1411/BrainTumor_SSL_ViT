import sys
import os

# 1. Định vị thư mục gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml

from models.classifier import BrainTumorClassifier as ViTClassifier

def evaluate():
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU TEST...")
    
    # Read config
    config_path = "configs/finetune.yaml"
    if not os.path.exists(config_path):
        print("Không tìm thấy configs/finetune.yaml")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đánh giá: {device}")

    # DataLoader Test Set
    data_dir = config["dataset"]["data_dir"]
    test_dir = os.path.join(data_dir, "Testing") if os.path.exists(os.path.join(data_dir, "Testing")) else data_dir

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    class_names = test_dataset.classes
    print(f" Các lớp u nào ({len(class_names)}): {class_names}")

    # Init model & Load Best Weights
    model = ViTClassifier(num_classes=len(class_names), freeze_backbone=False).to(device)
    
    weights_path = "checkpoints/finetune/best_classifier.pth"
    if not os.path.exists(weights_path):
        print(f" Không tìm thấy file trọng số tại: {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Đã nạp thành công trọng số từ: {weights_path}")

    # Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Print Classification Report
    print("\n" + "="*60)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CHI TIẾT (CLASSIFICATION REPORT)")
    print("="*60)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report)

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Brain Tumor Classification (SSL + ViT)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    save_cm_path = "results/confusion_matrix.png"
    plt.savefig(save_cm_path, dpi=300)
    print(f"Đã lưu đồ thị Ma trận Nhầm lẫn tại: {save_cm_path}")
    print("="*60)

if __name__ == "__main__":
    evaluate()