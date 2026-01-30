"""
Utility functions for data preprocessing and augmentation.
"""

import cv2
import numpy as np
from PIL import Image

def resize_image(image, target_size=(640, 640)):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Tuple of (width, height)
    
    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create new image with padding
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_size[0] - image.width) // 2
    paste_y = (target_size[1] - image.height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    return np.array(new_image)

def crop_face(image, bbox, margin=0.2):
    """
    Crop face region from image with margin.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Extra margin around face (proportion)
    
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = bbox
    
    # Add margin
    width = x2 - x1
    height = y2 - y1
    
    x1 = max(0, int(x1 - width * margin))
    y1 = max(0, int(y1 - height * margin))
    x2 = min(image.shape[1], int(x2 + width * margin))
    y2 = min(image.shape[0], int(y2 + height * margin))
    
    return image[y1:y2, x1:x2]

def apply_histogram_equalization(image):
    """Apply histogram equalization to improve contrast."""
    if len(image.shape) == 3:
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def normalize_image(image):
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0
