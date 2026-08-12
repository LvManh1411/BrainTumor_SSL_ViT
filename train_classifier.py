import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import get_dataloaders
from models.classifier import BrainTumorClassifier
from trainer.finetune import FineTuneTrainer
def main():
    # 1. Đọc config
    with open("configs/finetune.yaml", "r", encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")
    # 2. Load Dataloaders (Train, Val, Test)
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        img_size=config["dataset"]["img_size"]
    )
    # 3. Khởi tạo Model Classifier
    model = BrainTumorClassifier(
        num_classes=config["model"]["num_classes"],
        ssl_checkpoint_path=config["model"]["ssl_checkpoint"],
        freeze_backbone=config["model"]["freeze_backbone"]
    ).to(device)

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["train"]["learning_rate"], 
        weight_decay=config["train"]["weight_decay"]
    )
    # 5. Khởi chạy Trainer
    trainer = FineTuneTrainer(model, train_loader, val_loader, optimizer, criterion, device, config)
    trainer.fit()

if __name__ == "__main__":
    main()