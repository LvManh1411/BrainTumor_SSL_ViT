import sys
import os

# Bổ sung thư mục gốc dự án vào sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Fix trùng tên package: Thử import từ root, nếu chạy trực tiếp thì import local
try:
    from dataset.transforms import get_train_transforms, get_val_transforms
except (ModuleNotFoundError, ImportError):
    from transforms import get_train_transforms, get_val_transforms


def get_dataloaders(data_dir="data/raw", batch_size=32, num_workers=2, img_size=224):
    """
    Khởi tạo DataLoaders kết nối trực tiếp với file transforms.py
    """
    train_transform = get_train_transforms(img_size)
    test_transform = get_val_transforms(img_size)
    
    train_path = os.path.join(data_dir, "Training")
    test_path = os.path.join(data_dir, "Testing")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f" Không tìm thấy thư mục Training tại: {train_path}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform) if os.path.exists(test_path) else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    ) if test_dataset else None
    
    return train_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
    
    train_loader, test_loader, classes = get_dataloaders(data_dir=RAW_DATA_PATH)
    print(f" Các lớp u tìm thấy ({len(classes)}): {classes}")
    
    images, labels = next(iter(train_loader))
    print(f" Kích thước 1 Batch ảnh Train (có Augmentation): {images.shape}")
    print(f" Kích thước 1 Batch Nhãn (Labels): {labels.shape}")