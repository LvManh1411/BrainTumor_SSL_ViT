import sys
import os
import random

# Bổ sung thư mục gốc dự án vào sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from PIL import Image

# Fix trùng tên package: Thử import từ root, nếu chạy trực tiếp thì import local
try:
    from dataset.transforms import get_train_transforms, get_val_transforms
except (ModuleNotFoundError, ImportError):
    from transforms import get_train_transforms, get_val_transforms


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


# =========================================================
# 1. SEED CỐ ĐỊNH - đảm bảo khả năng tái lập kết quả
# =========================================================

def set_seed(seed=42):
    """
    Cố định toàn bộ nguồn ngẫu nhiên (Python random, NumPy, PyTorch CPU/GPU)
    để đảm bảo mỗi lần chạy cho ra kết quả giống nhau - yêu cầu bắt buộc
    khi báo cáo kết quả thực nghiệm khoa học.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Đảm bảo các phép toán CUDA cũng deterministic (có thể làm chậm nhẹ
    # tốc độ train, nhưng cần thiết để tái lập kết quả chính xác)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2. DATASET CÓ NHÃN (Training/Testing) - dùng cho Fine-tuning (Nội dung 3)
# =========================================================

def get_dataloaders(data_dir="data/processed", batch_size=32, num_workers=2,
                     img_size=224, val_ratio=0.15, seed=42):
    """
    Khởi tạo DataLoaders cho bài toán phân loại có giám sát (Fine-tuning ViT).
    Trả về 3 tập: train / val / test - đúng yêu cầu trong thuyết minh đề tài.

    - train: có Augmentation (get_train_transforms)
    - val, test: KHÔNG Augmentation, chỉ Resize + Normalize (get_val_transforms)
    - val được tách ra từ Training theo val_ratio, dùng seed cố định để
      lần nào chạy cũng chia đúng như vậy (tái lập được)
    """
    set_seed(seed)

    train_transform = get_train_transforms(img_size)
    eval_transform = get_val_transforms(img_size)

    train_path = os.path.join(data_dir, "Training")
    test_path = os.path.join(data_dir, "Testing")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục Training tại: {train_path}")

    # Load 2 lần với transform khác nhau, để val KHÔNG bị augment
    # (augment chỉ nên áp dụng cho tập train)
    full_train_aug = datasets.ImageFolder(root=train_path, transform=train_transform)
    full_train_eval = datasets.ImageFolder(root=train_path, transform=eval_transform)

    n_total = len(full_train_aug)
    n_val = int(n_total * val_ratio)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    test_dataset = (
        datasets.ImageFolder(root=test_path, transform=eval_transform)
        if os.path.exists(test_path) else None
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True)
        if test_dataset else None
    )

    print(f"Dataset: {n_total} ảnh Training gốc -> "
          f"{len(train_indices)} train / {len(val_indices)} val "
          f"(val_ratio={val_ratio}, seed={seed})")
    if test_dataset:
        print(f"Testing: {len(test_dataset)} ảnh")

    return train_loader, val_loader, test_loader, full_train_aug.classes


# =========================================================
# 3. DATASET KHÔNG NHÃN - dùng cho SSL Pre-training (Nội dung 2)
# =========================================================

class UnlabeledMRIDataset(Dataset):
    """
    Dataset cho ảnh MRI CHƯA GÁN NHÃN, dùng cho giai đoạn tiền huấn luyện
    Self-Supervised Learning (SSL). Khác với ImageFolder (yêu cầu cấu trúc
    thư mục theo class), dataset này đọc TOÀN BỘ ảnh trong 1 thư mục gốc,
    bất kể có phân theo lớp hay không, và KHÔNG trả về nhãn.

    Nhiều thuật toán SSL (SimCLR, DINO...) cần 2 ảnh augment khác nhau từ
    CÙNG 1 ảnh gốc để học biểu diễn tương phản (contrastive learning) ->
    hỗ trợ tham số n_views để trả về nhiều bản augment của cùng 1 ảnh.
    """

    def __init__(self, root_dir, transform=None, n_views=1):
        """
        Args:
            root_dir: thư mục gốc chứa ảnh (ví dụ data/unlabeled/), quét
                      đệ quy toàn bộ ảnh trong mọi thư mục con.
            transform: augmentation áp dụng cho từng ảnh (nên dùng
                       transform có augment mạnh nếu dùng SimCLR/DINO).
            n_views: số bản augment trả về cho mỗi ảnh gốc. n_views=1 phù
                     hợp cho MAE (chỉ cần 1 view để mask); n_views=2 phù
                     hợp cho SimCLR/DINO (cần cặp view để so sánh tương phản).
        """
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(VALID_EXTENSIONS):
                    self.image_paths.append(os.path.join(dirpath, f))

        if len(self.image_paths) == 0:
            print(f"[Cảnh báo] Không tìm thấy ảnh nào trong {root_dir}")

        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is None:
            return img

        if self.n_views == 1:
            return self.transform(img)

        # Trả về nhiều view augment khác nhau của CÙNG 1 ảnh gốc
        # (dùng cho SimCLR/DINO - cần cặp/bộ view để học contrastive)
        return [self.transform(img) for _ in range(self.n_views)]


def get_unlabeled_dataloader(data_dir="data/unlabeled", batch_size=64,
                              num_workers=2, img_size=224, n_views=1, seed=42):
    """
    Khởi tạo DataLoader cho giai đoạn SSL Pre-training (Nội dung 2).
    Dùng transform có augmentation (get_train_transforms) vì hầu hết thuật
    toán SSL dựa vào augmentation mạnh để tạo tín hiệu học tự giám sát.
    """
    set_seed(seed)

    transform = get_train_transforms(img_size)
    dataset = UnlabeledMRIDataset(root_dir=data_dir, transform=transform, n_views=n_views)

    if len(dataset) == 0:
        raise ValueError(
            f"Không tìm thấy ảnh nào trong '{data_dir}'. "
            f"Hãy chuẩn bị dữ liệu MRI chưa gán nhãn (Nội dung 1-2 trong thuyết minh) "
            f"trước khi chạy SSL pretrain."
        )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"Unlabeled dataset (SSL pretrain): {len(dataset)} ảnh, n_views={n_views}")
    return loader


# =========================================================
# TEST NHANH
# =========================================================

if __name__ == "__main__":
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
    UNLABELED_DATA_PATH = os.path.join(BASE_DIR, "data", "unlabeled")

    print("=" * 60)
    print("TEST 1: Dataloaders có nhãn (train/val/test) - cho Fine-tuning")
    print("=" * 60)
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=PROCESSED_DATA_PATH
    )
    print(f"Các lớp u tìm thấy ({len(classes)}): {classes}")

    images, labels = next(iter(train_loader))
    print(f"Batch Train: ảnh {images.shape} | nhãn {labels.shape}")

    images_val, labels_val = next(iter(val_loader))
    print(f"Batch Val:   ảnh {images_val.shape} | nhãn {labels_val.shape}")

    if test_loader:
        images_test, labels_test = next(iter(test_loader))
        print(f"Batch Test:  ảnh {images_test.shape} | nhãn {labels_test.shape}")

    print("\n" + "=" * 60)
    print("TEST 2: Dataloader KHÔNG nhãn - cho SSL Pre-training")
    print("=" * 60)

    def count_images_recursive(root_dir):
        if not os.path.exists(root_dir):
            return 0
        count = 0
        for _, _, filenames in os.walk(root_dir):
            count += sum(1 for f in filenames if f.lower().endswith(VALID_EXTENSIONS))
        return count

    n_unlabeled_images = count_images_recursive(UNLABELED_DATA_PATH)

    if n_unlabeled_images > 0:
        unlabeled_loader = get_unlabeled_dataloader(
            data_dir=UNLABELED_DATA_PATH, n_views=1
        )
        batch = next(iter(unlabeled_loader))
        print(f"Batch Unlabeled: {batch.shape}")
    else:
        print(f"[Bỏ qua] Thư mục {UNLABELED_DATA_PATH} chưa có ảnh (0 ảnh tìm thấy) - "
              f"hãy chuẩn bị dữ liệu chưa gán nhãn trước khi chạy SSL pretrain.")