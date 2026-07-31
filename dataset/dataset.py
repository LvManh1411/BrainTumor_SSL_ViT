import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Định nghĩa Phép biến đổi ảnh (Transforms)
def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# 2. Hàm khởi tạo DataLoaders
def get_dataloaders(data_dir="data/raw", batch_size=32, num_workers=2, img_size=224):
    train_transform, test_transform = get_transforms(img_size)
    
    train_path = os.path.join(data_dir, "Training")
    test_path = os.path.join(data_dir, "Testing")
    
    # Load dataset tự động từ thư mục theo class
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes

# --- ĐOẠN CODE TEST NHANH ---
if __name__ == "__main__":
    # Tự động lấy đường dẫn tuyệt đối tới thư mục gốc dự án
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
    
    train_loader, test_loader, classes = get_dataloaders(data_dir=RAW_DATA_PATH)
    print(f"Các lớp u tìm thấy ({len(classes)}): {classes}")
    
    images, labels = next(iter(train_loader))
    print(f"Kích thước 1 Batch ảnh Train: {images.shape}")
    print(f"Kích thước 1 Batch Nhãn (Labels): {labels.shape}")