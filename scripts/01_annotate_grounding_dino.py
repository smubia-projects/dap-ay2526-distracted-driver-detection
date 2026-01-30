"""
Grounding DINO Auto-Annotation Script
Automatically annotates images with bounding boxes using Grounding DINO.
Outputs annotations in YOLO format.
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Grounding DINO imports
# Note: You'll need to install GroundingDINO first
try:
    from groundingdino.util.inference import load_model, load_image, predict
    from groundingdino.util import box_ops
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    print("WARNING: GroundingDINO not installed. This script will not work.")
    print("Install from: https://github.com/IDEA-Research/GroundingDINO")
    GROUNDING_DINO_AVAILABLE = False

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_grounding_dino(config):
    """Initialize Grounding DINO model."""
    model_config = config['grounding_dino']['model_config']
    checkpoint = config['grounding_dino']['checkpoint']
    device = config['grounding_dino']['device']
    
    print(f"Loading Grounding DINO model from {checkpoint}...")
    model = load_model(model_config, checkpoint, device=device)
    return model

def process_image(model, image_path, text_prompt, box_threshold, text_threshold):
    """
    Process a single image with Grounding DINO.
    
    Returns:
        boxes: Bounding boxes in normalized [x_center, y_center, width, height] format
        logits: Confidence scores
        phrases: Detected object labels
    """
    # Load image
    image_source, image = load_image(image_path)
    
    # Run inference
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height]
    h, w = image_source.shape[:2]
    boxes_normalized = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        boxes_normalized.append([x_center, y_center, width, height])
    
    return boxes_normalized, logits, phrases

def save_yolo_annotation(boxes, labels, output_path, class_mapping):
    """
    Save annotations in YOLO format.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    with open(output_path, 'w') as f:
        for box, label in zip(boxes, labels):
            # Map label to class ID (you may need to customize this)
            class_id = class_mapping.get(label.lower(), 0)
            
            x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def annotate_dataset(model, config, split='train'):
    """
    Annotate all images in a dataset split.
    
    Args:
        model: Grounding DINO model
        config: Configuration dictionary
        split: Dataset split ('train', 'val', or 'test')
    """
    # Paths
    splits_dir = config['data']['splits_dir']
    annotated_dir = config['data']['annotated_dir']
    
    input_dir = os.path.join(splits_dir, split)
    output_dir = os.path.join(annotated_dir, split)
    
    # Grounding DINO parameters
    text_prompt = config['grounding_dino']['text_prompts'][0]
    box_threshold = config['grounding_dino']['box_threshold']
    text_threshold = config['grounding_dino']['text_threshold']
    
    # Create label mapping for detected objects
    label_mapping = {
        'driver': 0,
        'face': 1,
        'hand': 2,
        'hands': 2,
        'steering wheel': 3,
        'phone': 4,
        'cup': 5,
    }
    
    print(f"\nAnnotating {split} set...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Text prompt: '{text_prompt}'")
    
    # Get all image files
    image_files = []
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append((class_dir, img_file))
    
    print(f"Found {len(image_files)} images to annotate\n")
    
    # Process each image
    for class_name, img_file in tqdm(image_files, desc=f"Annotating {split}"):
        # Input path
        img_path = os.path.join(input_dir, class_name, img_file)
        
        # Output paths
        output_class_dir = os.path.join(output_dir, 'images', class_name)
        output_labels_dir = os.path.join(output_dir, 'labels', class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        try:
            # Run Grounding DINO
            boxes, logits, phrases = process_image(
                model, img_path, text_prompt, box_threshold, text_threshold
            )
            
            # Save annotation
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_file)
            save_yolo_annotation(boxes, phrases, label_path, label_mapping)
            
            # Copy image to output directory
            import shutil
            shutil.copy2(img_path, os.path.join(output_class_dir, img_file))
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

def main():
    if not GROUNDING_DINO_AVAILABLE:
        print("\nPlease install GroundingDINO before running this script:")
        print("git clone https://github.com/IDEA-Research/GroundingDINO.git")
        print("cd GroundingDINO && pip install -e .")
        return
    
    # Load configuration
    config = load_config()
    
    print("=" * 60)
    print("Grounding DINO Auto-Annotation")
    print("=" * 60)
    
    # Setup model
    model = setup_grounding_dino(config)
    
    # Annotate each split
    for split in ['train', 'val', 'test']:
        annotate_dataset(model, config, split)
    
    print("\n" + "=" * 60)
    print("Annotation complete!")
    print("=" * 60)
    print("\nNext step: python scripts/02_prepare_yolo_data.py")

if __name__ == "__main__":
    main()
