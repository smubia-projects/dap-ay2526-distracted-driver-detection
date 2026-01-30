"""
Data Splitting Script
Splits the raw training data into train/validation/test sets with stratification.
"""

import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from tqdm import tqdm

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_all_images(data_dir):
    """
    Get all images organized by class.
    
    Returns:
        dict: {class_name: [list of image paths]}
    """
    images_by_class = defaultdict(list)
    
    for class_dir in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    images_by_class[class_dir].append(img_path)
    
    return images_by_class

def split_data(images_by_class, train_ratio, val_ratio, test_ratio, random_seed=42):
    """
    Split data into train/val/test sets with stratification.
    
    Args:
        images_by_class: Dictionary of {class: [image_paths]}
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    random.seed(random_seed)
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_name, images in images_by_class.items():
        # Shuffle images
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Add class information to each image path
        for img in train_imgs:
            splits['train'].append((img, class_name))
        for img in val_imgs:
            splits['val'].append((img, class_name))
        for img in test_imgs:
            splits['test'].append((img, class_name))
        
        print(f"Class {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    return splits

def copy_files(splits, output_dir):
    """
    Copy files to their respective split directories.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Base output directory
    """
    for split_name, images in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        
        print(f"\nCopying {len(images)} images to {split_name} set...")
        
        for img_path, class_name in tqdm(images):
            # Create class directory in split
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Copy file
            img_filename = os.path.basename(img_path)
            dest_path = os.path.join(class_dir, img_filename)
            shutil.copy2(img_path, dest_path)

def create_label_mapping(config, output_dir):
    """Create a label mapping file for reference."""
    class_mapping = config['data']['classes']
    
    mapping_file = os.path.join(output_dir, 'class_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("Class ID -> Class Name\n")
        f.write("=" * 50 + "\n")
        for class_id, class_name in sorted(class_mapping.items()):
            f.write(f"{class_id} -> {class_name}\n")
    
    print(f"\nClass mapping saved to: {mapping_file}")

def main():
    # Load configuration
    config = load_config()
    
    # Set random seed
    random_seed = config['project']['random_seed']
    
    # Paths
    raw_data_dir = config['data']['raw_dir']
    output_dir = config['data']['splits_dir']
    
    # Split ratios
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    print("=" * 60)
    print("Driver Distraction Dataset - Data Splitting")
    print("=" * 60)
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    print(f"Random seed: {random_seed}\n")
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Get all images
    print("Scanning raw data directory...")
    images_by_class = get_all_images(raw_data_dir)
    
    total_images = sum(len(imgs) for imgs in images_by_class.values())
    print(f"\nFound {total_images} images across {len(images_by_class)} classes\n")
    
    # Split data
    print("Splitting data with stratification...")
    splits = split_data(images_by_class, train_ratio, val_ratio, test_ratio, random_seed)
    
    print(f"\nTotal splits:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")
    
    # Copy files to split directories
    copy_files(splits, output_dir)
    
    # Create label mapping file
    create_label_mapping(config, output_dir)
    
    print("\n" + "=" * 60)
    print("Data splitting complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Run Grounding DINO annotation: python scripts/01_annotate_grounding_dino.py")
    print(f"2. Prepare YOLO data: python scripts/02_prepare_yolo_data.py")

if __name__ == "__main__":
    main()
