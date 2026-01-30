"""
OpenPose Pose Extraction Script
Extract body and head pose keypoints for drowsiness detection.
Uses MediaPipe as a lighter alternative to OpenPose.
"""

import os
import yaml
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class PoseExtractor:
    """Extract pose keypoints using MediaPipe."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe Face Mesh for detailed face landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def extract_pose(self, image_path):
        """
        Extract pose keypoints from an image.
        
        Returns:
            dict: Pose information including keypoints, head angle, etc.
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process with pose detection
        pose_results = self.pose.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)
        
        pose_data = {
            'image_path': image_path,
            'image_size': [w, h],
            'pose_landmarks': [],
            'face_landmarks': [],
            'head_angle': None,
            'body_posture': None
        }
        
        # Extract pose landmarks
        if pose_results.pose_landmarks:
            landmarks = []
            for landmark in pose_results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            pose_data['pose_landmarks'] = landmarks
            
            # Calculate head angle (tilt)
            pose_data['head_angle'] = self.calculate_head_angle(landmarks)
            
            # Analyze body posture
            pose_data['body_posture'] = self.analyze_posture(landmarks)
        
        # Extract face landmarks
        if face_results.multi_face_landmarks:
            face_landmarks = []
            for landmark in face_results.multi_face_landmarks[0].landmark:
                face_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            pose_data['face_landmarks'] = face_landmarks
        
        return pose_data
    
    def calculate_head_angle(self, landmarks):
        """
        Calculate head tilt angle from pose landmarks.
        
        Uses nose and shoulders to estimate head angle.
        """
        if len(landmarks) < 12:
            return None
        
        # Get key points (MediaPipe Pose landmarks)
        # 0: nose, 2: left eye, 5: right eye, 11: left shoulder, 12: right shoulder
        nose = np.array([landmarks[0]['x'], landmarks[0]['y']])
        left_shoulder = np.array([landmarks[11]['x'], landmarks[11]['y']])
        right_shoulder = np.array([landmarks[12]['x'], landmarks[12]['y']])
        
        # Calculate shoulder midpoint
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        
        # Calculate angle
        diff = nose - shoulder_midpoint
        angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi
        
        return float(angle)
    
    def analyze_posture(self, landmarks):
        """
        Analyze body posture for drowsiness indicators.
        
        Returns:
            str: 'upright', 'leaning_left', 'leaning_right', 'slouching'
        """
        if len(landmarks) < 24:
            return 'unknown'
        
        # Get shoulder and hip positions
        left_shoulder = np.array([landmarks[11]['y']])
        right_shoulder = np.array([landmarks[12]['y']])
        left_hip = np.array([landmarks[23]['y']])
        right_hip = np.array([landmarks[24]['y']])
        
        # Calculate shoulder and hip tilt
        shoulder_tilt = left_shoulder - right_shoulder
        hip_tilt = left_hip - right_hip
        
        # Determine posture
        if abs(shoulder_tilt) < 0.05:
            return 'upright'
        elif shoulder_tilt > 0.05:
            return 'leaning_right'
        elif shoulder_tilt < -0.05:
            return 'leaning_left'
        else:
            return 'unknown'
    
    def visualize_pose(self, image_path, pose_data, output_path):
        """Visualize pose keypoints on the image."""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw pose landmarks
        if pose_data['pose_landmarks']:
            # Convert back to MediaPipe format for drawing
            pose_results = self.pose.process(image_rgb)
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
        
        # Add text annotations
        if pose_data['head_angle'] is not None:
            cv2.putText(image, f"Head Angle: {pose_data['head_angle']:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if pose_data['body_posture']:
            cv2.putText(image, f"Posture: {pose_data['body_posture']}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, image)

def process_dataset(config, split='train', visualize=False):
    """
    Process all images in a dataset split to extract poses.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', or 'test')
        visualize: Whether to save visualizations
    """
    splits_dir = config['data']['splits_dir']
    preprocessed_dir = config['data']['preprocessed_dir']
    
    input_dir = os.path.join(splits_dir, split)
    output_dir = os.path.join(preprocessed_dir, 'poses', split)
    os.makedirs(output_dir, exist_ok=True)
    
    if visualize:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nExtracting poses from {split} set...")
    
    # Initialize pose extractor
    extractor = PoseExtractor(config)
    
    # Get all image files
    image_files = []
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append((class_dir, img_file))
    
    # Process each image
    all_poses = []
    for class_name, img_file in tqdm(image_files, desc=f"Processing {split}"):
        img_path = os.path.join(input_dir, class_name, img_file)
        
        try:
            # Extract pose
            pose_data = extractor.extract_pose(img_path)
            
            if pose_data:
                pose_data['class'] = class_name
                all_poses.append(pose_data)
                
                # Save visualization if requested
                if visualize and pose_data['pose_landmarks']:
                    viz_path = os.path.join(viz_dir, f"{class_name}_{img_file}")
                    extractor.visualize_pose(img_path, pose_data, viz_path)
        
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    # Save pose data
    output_file = os.path.join(output_dir, f'{split}_poses.json')
    with open(output_file, 'w') as f:
        json.dump(all_poses, f, indent=2)
    
    print(f"Saved {len(all_poses)} pose annotations to {output_file}")
    
    return all_poses

def main():
    # Load configuration
    config = load_config()
    
    print("=" * 60)
    print("Pose Extraction with MediaPipe")
    print("=" * 60)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        process_dataset(config, split, visualize=True)
    
    print("\n" + "=" * 60)
    print("Pose extraction complete!")
    print("=" * 60)
    print("\nNext step: python scripts/05_train_resnet.py")

if __name__ == "__main__":
    main()
