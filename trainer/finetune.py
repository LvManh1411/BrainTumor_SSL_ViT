import os
import torch
import torch.nn as nn

class FineTuneTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.save_dir = config["train"]["save_dir"]
        
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, acc

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, acc

    def fit(self):
        print("\nBẮT ĐẦU QUÁ TRÌNH FINE-TUNING PHÂN LOẠI U NÃO...")
        best_acc = 0.0
        
        for epoch in range(self.config["train"]["epochs"]):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            print(f"Epoch [{epoch+1}/{self.config['train']['epochs']}] "
                  f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                  f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(self.save_dir, "best_classifier.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Đã lưu Best Model với Val Acc: {best_acc:.2f}%")