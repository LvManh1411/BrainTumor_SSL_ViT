import torchvision.transforms as T

def get_train_transforms(img_size=224):
    """
    Transform cho tập TRAINING: Có thêm Data Augmentation để tăng accuracy
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),              # Lật ngang ngẫu nhiên
        T.RandomRotation(degrees=15),                # Xoay ngẫu nhiên ±15 độ
        T.ColorJitter(brightness=0.1, contrast=0.1),# Chỉnh nhẹ độ sáng/tương phản
        T.ToTensor(),                                # Chuyển ảnh về Tensor
        T.Normalize(                                 # Chuẩn hóa ImageNet
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(img_size=224):
    """
    Transform cho tập VAL / TEST / DEMO WEB: Chỉ Resize và Chuẩn hóa dữ liệu gốc
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])