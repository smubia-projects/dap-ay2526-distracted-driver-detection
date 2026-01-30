"""
ResNet Training Script for Drowsiness/Distraction Detection
Fine-tune ResNet on cropped face images to detect driver states.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DriverDataset(Dataset):
    """Dataset for driver distraction/drowsiness images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory with class subdirectories (c0, c1, ...)
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Scan directory for images
        for class_dir in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                class_idx = int(class_dir[1:])  # Extract number from 'c0', 'c1', etc.
                self.class_to_idx[class_dir] = class_idx
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms(config):
    """Get data augmentation and normalization transforms."""
    img_size = config['data']['image_size'][0]
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(config, num_classes=10):
    """
    Create ResNet model for classification.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    resnet_config = config['resnet']
    architecture = resnet_config['architecture']
    pretrained = resnet_config['pretrained']
    
    # Load pretrained ResNet
    if architecture == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif architecture == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif architecture == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")

def train_model(config):
    """Main training function."""
    # Configuration
    resnet_config = config['resnet']
    splits_dir = config['data']['splits_dir']
    
    epochs = resnet_config['epochs']
    batch_size = resnet_config['batch_size']
    learning_rate = resnet_config['learning_rate']
    weight_decay = resnet_config['weight_decay']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data transforms
    train_transform, val_transform = get_data_transforms(config)
    
    # Datasets
    train_dir = os.path.join(splits_dir, 'train')
    val_dir = os.path.join(splits_dir, 'val')
    test_dir = os.path.join(splits_dir, 'test')
    
    train_dataset = DriverDataset(train_dir, transform=train_transform)
    val_dataset = DriverDataset(val_dir, transform=val_transform)
    test_dataset = DriverDataset(test_dir, transform=val_transform)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images\n")
    
    # Model
    num_classes = 10  # c0 to c9
    model = create_model(config, num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                          weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = 'models/resnet/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, best_model_path)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Test on best model
    print("\n" + "=" * 60)
    print("Testing Best Model")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%\n")
    
    # Classification report
    class_names = [config['data']['classes'][f'c{i}'] for i in range(10)]
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                               target_names=class_names))
    
    # Confusion matrix
    cm_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {best_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

def main():
    # Load configuration
    config = load_config()
    
    # Train model
    train_model(config)
    
    print("\nNext step: python scripts/pipeline.py")

if __name__ == "__main__":
    main()
