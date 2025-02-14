import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data_utils import ACDCDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_model(model, optimizer, epoch, loss, save_path="resnet152_acdc.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved at epoch {epoch} to {save_path}")

# Training setup
def train_model(train_dir, test_dir, num_epochs=50, batch_size=16, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/acdc_resnet152")

    # Load datasets
    train_dataset = ACDCDataset(train_dir,transform=None)
    test_dataset = ACDCDataset(test_dir,transform=None)
    
    #unique_labels = set(train_dataset.patient_labels.values())
    #print(f"Unique labels in dataset: {unique_labels}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load ResNet-152
    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)  # 4 classes in ACDC
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_acc = 0.0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        scheduler.step(train_acc)
    
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2%}")
        if train_acc > best_acc:
            best_acc = train_acc
            save_model(model, optimizer, epoch, train_acc, "models/best_resnet152_acdc.pth")
    
    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_acc = correct / total
    writer.add_scalar("Accuracy/test", test_acc, num_epochs)
    print(f"Test Accuracy: {test_acc:.2%}")
    writer.close()

if __name__ == "__main__":
    train_model("/mnt/data/ACDC/training/", "/mnt/data/ACDC/testing/", num_epochs=50, batch_size=128)
