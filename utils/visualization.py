"""
Visualization utilities for model outputs and results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def draw_bboxes(image, bboxes, labels=None, confidences=None, color=(0, 255, 0)):
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        labels: Optional list of labels
        confidences: Optional list of confidence scores
        color: Box color (B, G, R)
    
    Returns:
        Image with drawn bounding boxes
    """
    annotated = image.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        if labels is not None:
            label = labels[i]
            if confidences is not None:
                label = f"{label}: {confidences[i]:.2f}"
            
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated

def plot_training_history(history, output_path):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_class_distribution_plot(class_counts, class_names, output_path):
    """
    Create bar plot of class distribution.
    
    Args:
        class_counts: Dictionary or list of counts per class
        class_names: List of class names
        output_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(class_counts, dict):
        counts = [class_counts.get(name, 0) for name in class_names]
    else:
        counts = class_counts
    
    plt.bar(range(len(class_names)), counts, color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_sample_predictions(images, true_labels, pred_labels, class_names, output_path, n_samples=16):
    """
    Save grid of sample predictions.
    
    Args:
        images: List of images
        true_labels: True class labels
        pred_labels: Predicted class labels
        class_names: List of class names
        output_path: Path to save visualization
        n_samples: Number of samples to display
    """
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Display image
        img = images[i]
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        # Denormalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        
        # Title
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f"True: {true_class}\nPred: {pred_class}", color=color, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
