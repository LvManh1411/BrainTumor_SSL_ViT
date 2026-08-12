import yaml
import torch
import torch.optim as optim
from dataset.dataset import get_dataloaders
from models.ssl_model import SSLViT
from trainer.ssl_trainer import SSLTrainer

def main():
    # 1. Đọc file cấu hình
    with open("configs/ssl.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Đang sử dụng thiết bị: {device}")

    # 2. Nạp dữ liệu
    train_loader, _, _ = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        img_size=config["dataset"]["img_size"]
    )

    # 3. Khởi tạo mô hình & Optimizer
    model = SSLViT(
        pretrained=config["model"]["pretrained"],
        projection_dim=config["model"]["projection_dim"]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["train"]["learning_rate"], 
        weight_decay=config["train"]["weight_decay"]
    )

    # 4. Kích hoạt Trainer
    trainer = SSLTrainer(model, train_loader, optimizer, device, config)
    trainer.fit()

if __name__ == "__main__":
    main()