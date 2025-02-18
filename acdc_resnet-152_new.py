import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data_utils import ACDCDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
import random

def save_model(model, optimizer, epoch, loss, save_path="resnet152_acdc_new.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved at epoch {epoch} to {save_path}")

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

# MixUp Augmentation

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(train_dir, test_dir, num_epochs=50, batch_size=32, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/acdc_resnet152")

    # Data Augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load datasets
    train_dataset = ACDCDataset(train_dir, transform=transform)
    test_dataset = ACDCDataset(test_dir, transform=transforms.ToTensor())
    
    # Class balancing
    class_counts = np.bincount([train_dataset.label_mapping[label] for label in train_dataset.patient_labels.values()])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[train_dataset.label_mapping[label]] for label in train_dataset.patient_labels.values()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load ResNet-152
    model = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2')
    
    # Unfreeze the last block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2%}")
        if train_acc > best_acc:
            best_acc = train_acc
            save_model(model, optimizer, epoch, train_acc, "models/best_resnet152_acdc.pth")
    
    
    # Define Test-Time Augmentations (TTA)
    tta_transforms = [
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    ]

    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:  
            images, labels = images.to(device), labels.to(device)
            batch_preds = []

            # Apply test-time augmentations
            for trans in tta_transforms:
                transformed_images = torch.stack([trans(img.cpu()) for img in images])  
                transformed_images = transformed_images.to(device)
                preds = model(transformed_images)  
                batch_preds.append(preds)

            # Average predictions across TTA versions
            batch_preds = torch.stack(batch_preds).mean(dim=0)  

            # Compute accuracy
            _, predicted = torch.max(batch_preds, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted)
            all_labels.append(labels)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")
    writer.add_scalar("Accuracy/test", accuracy, num_epochs)
    writer.close()

if __name__ == "__main__":
    train_model("/mnt/data/ACDC/training/", "/mnt/data/ACDC/testing/", num_epochs=50, batch_size=32)
