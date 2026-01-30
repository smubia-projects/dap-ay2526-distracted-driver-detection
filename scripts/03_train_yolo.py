"""
YOLO Model Training Script
Train YOLOv8 for driver and object detection.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_yolo(config):
    """
    Train YOLO model for object detection.
    
    Detects: driver, face, hands, steering wheel, phone, cup
    """
    yolo_config = config['yolo']
    
    # Model configuration
    model_name = yolo_config['model']
    epochs = yolo_config['epochs']
    batch_size = yolo_config['batch_size']
    img_size = yolo_config['img_size']
    device = yolo_config['device']
    
    # Dataset configuration
    dataset_yaml = 'models/yolo/config/dataset.yaml'
    
    # Output directory
    output_dir = 'models/yolo/weights'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("YOLOv8 Training - Object Detection")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Load a pretrained model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name='driver_detection',
        
        # Hyperparameters
        lr0=yolo_config['lr0'],
        lrf=yolo_config['lrf'],
        momentum=yolo_config['momentum'],
        weight_decay=yolo_config['weight_decay'],
        
        # Augmentation
        hsv_h=yolo_config['hsv_h'],
        hsv_s=yolo_config['hsv_s'],
        hsv_v=yolo_config['hsv_v'],
        degrees=yolo_config['degrees'],
        translate=yolo_config['translate'],
        scale=yolo_config['scale'],
        shear=yolo_config['shear'],
        perspective=yolo_config['perspective'],
        flipud=yolo_config['flipud'],
        fliplr=yolo_config['fliplr'],
        mosaic=yolo_config['mosaic'],
        
        # Other settings
        patience=yolo_config['patience'],
        workers=yolo_config['workers'],
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {output_dir}/driver_detection/weights/best.pt")
    print(f"Last model saved to: {output_dir}/driver_detection/weights/last.pt")
    
    return results

def validate_model(config):
    """Validate the trained model on the test set."""
    print("\n" + "=" * 60)
    print("Validating Model on Test Set")
    print("=" * 60)
    
    # Load best model
    best_model_path = 'models/yolo/weights/driver_detection/weights/best.pt'
    
    if not os.path.exists(best_model_path):
        print(f"Model not found: {best_model_path}")
        return
    
    model = YOLO(best_model_path)
    
    # Run validation
    dataset_yaml = 'models/yolo/config/dataset.yaml'
    results = model.val(data=dataset_yaml, split='test')
    
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

def main():
    # Load configuration
    config = load_config()
    
    # Train model
    train_yolo(config)
    
    # Validate model
    validate_model(config)
    
    print("\nNext step: python scripts/04_extract_poses.py")

if __name__ == "__main__":
    main()
