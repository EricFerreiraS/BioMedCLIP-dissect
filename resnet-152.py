import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from data_utils import ACDCDataset, MnMsDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_model(model, optimizer, epoch, loss, dataset_name, save_path="resnet152_acdc_scratch.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    save_path = f"models/best_resnet152_{dataset_name}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved at epoch {epoch} to {save_path}")

def calculate_class_distribution(dataset):
    class_counts = {}
    for _, label in dataset:
        label = label.item() if torch.is_tensor(label) else label
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

# Training setup
def train_model(train_dir, test_dir=None, num_epochs=50, batch_size=16, learning_rate=1e-4, dataset_name="ACDC", num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{dataset_name}_resnet152")

    transform_resnet152 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Dynamically load the appropriate dataset class based on `dataset_name`
    if dataset_name == "ACDC":
        DatasetClass = ACDCDataset  # Replace with other dataset classes if needed
    elif dataset_name == "MnMs":
        DatasetClass = MnMsDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load datasets
    # Load datasets and handle train-test split if `test_dir` is None
    full_dataset = DatasetClass(train_dir, transform=transform_resnet152)
    
    if test_dir is None:
        test_split_ratio = 0.3  # Use 30% of the data for testing
        test_size = int(len(full_dataset) * test_split_ratio)
        train_size = len(full_dataset) - test_size
        
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        print(f"Dataset split: {train_size} training samples and {test_size} testing samples.")
    else:
        train_dataset = full_dataset
        test_dataset = DatasetClass(test_dir, transform=transform_resnet152)

    train_class_distribution = calculate_class_distribution(train_dataset)
    #test_class_distribution = calculate_class_distribution(test_dataset)

    print("Train Class Distribution:", train_class_distribution)
    #print("Test Class Distribution:", test_class_distribution)
    
    # Compute class weights inversely proportional to class frequencies
    class_weights = {cls: 1.0 / count for cls, count in train_class_distribution.items()}

    # Create weights for each sample based on its class
    sample_weights = [class_weights[int(label)] for _, label in train_dataset]

    # Convert sample weights to a tensor
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Compute weights inversely proportional to class frequencies
    class_weights = torch.tensor([1.0 / train_class_distribution[c] for c in range(num_classes)])
    class_weights = class_weights.to(device)

    #unique_labels = set(train_dataset.patient_labels.values())
    #print(f"Unique labels in dataset: {unique_labels}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load ResNet-152
    #model = models.resnet152(weights=None)
    model = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2')

    #model.fc = nn.Linear(model.fc.in_features, 8)  # Match the original training setup
    
    #checkpoint = torch.load('models/best_resnet152_MnMs_scratch_img.pth', map_location=device)
    #model.load_state_dict(checkpoint["model_state_dict"])

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 5 classes in ACDC and 8 in MnMs

    '''
    for param in model.parameters():
        param.requires_grad = True
    '''
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    #for param in model.layer4.parameters():
    #       param.requires_grad = True

    for param in model.fc.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)#,weight=class_weights)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4,momentum=0.9)
    #optimizer = optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_acc = 0.0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            '''
            if i == 2:  # Print only the first 3 batches
                break
            else:
                unique_labels, counts = torch.unique(labels, return_counts=True)
                print(f"Batch {i}:")
                for label, count in zip(unique_labels, counts):
                    print(f"  Class {label.item()}: {count.item()} samples")
            '''
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
            save_model(model, optimizer, epoch+1, best_acc, dataset_name)

        # Early stopping condition: stop training if accuracy reaches 100%
        if train_acc == 1.0:
            print(f"Early stopping triggered at epoch {epoch+1}: Training accuracy reached 100%.")
            break
    
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
    #train_model("/mnt/data/ACDC/training/", "/mnt/data/ACDC/testing/", num_epochs=25, batch_size=32, dataset_name="ACDC", num_classes=5)
    train_model("/mnt/data/MnM2s/MnM2/",None, num_epochs=25, batch_size=32, dataset_name="MnMs", num_classes=8)
