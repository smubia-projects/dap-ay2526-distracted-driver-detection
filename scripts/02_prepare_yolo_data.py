"""
Prepare YOLO Training Data
Converts annotated data to YOLO format and creates dataset configuration.
"""

import os
import yaml
import shutil
from pathlib import Path

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_yolo_dataset_yaml(config, output_path='models/yolo/config/dataset.yaml'):
    """
    Create YOLO dataset configuration file.
    
    YOLO expects a YAML file with:
    - path: dataset root directory
    - train: training images directory
    - val: validation images directory
    - names: class names dictionary
    """
    annotated_dir = os.path.abspath(config['data']['annotated_dir'])
    
    # Define paths relative to dataset root
    dataset_config = {
        'path': annotated_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        
        # Class names for object detection (from Grounding DINO)
        'names': {
            0: 'driver',
            1: 'face',
            2: 'hands',
            3: 'steering_wheel',
            4: 'phone',
            5: 'cup'
        }
    }
    
    # Save configuration
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"YOLO dataset config saved to: {output_path}")
    return output_path

def verify_dataset_structure(config):
    """Verify that the annotated dataset has the correct structure."""
    annotated_dir = config['data']['annotated_dir']
    
    print("\nVerifying dataset structure...")
    
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(annotated_dir, split, 'images')
        labels_dir = os.path.join(annotated_dir, split, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"  ⚠ Missing: {images_dir}")
        else:
            n_images = sum(len(files) for _, _, files in os.walk(images_dir) 
                          if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files))
            print(f"  ✓ {split}/images: {n_images} images")
        
        if not os.path.exists(labels_dir):
            print(f"  ⚠ Missing: {labels_dir}")
        else:
            n_labels = sum(len(files) for _, _, files in os.walk(labels_dir) 
                          if any(f.endswith('.txt') for f in files))
            print(f"  ✓ {split}/labels: {n_labels} annotations")

def main():
    # Load configuration
    config = load_config()
    
    print("=" * 60)
    print("YOLO Data Preparation")
    print("=" * 60)
    
    # Verify dataset structure
    verify_dataset_structure(config)
    
    # Create YOLO dataset configuration
    dataset_yaml = create_yolo_dataset_yaml(config)
    
    print("\n" + "=" * 60)
    print("YOLO data preparation complete!")
    print("=" * 60)
    print(f"\nDataset config: {dataset_yaml}")
    print("\nNext step: python scripts/03_train_yolo.py")

if __name__ == "__main__":
    main()
