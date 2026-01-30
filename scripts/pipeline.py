"""
Full Inference Pipeline
Combines all models (YOLO, OpenPose/MediaPipe, ResNet) for real-time 
driver drowsiness and distraction detection.
"""

import os
import yaml
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import argparse

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DriverMonitoringSystem:
    """
    Multi-model driver monitoring system combining:
    - YOLO for object detection
    - MediaPipe for pose estimation
    - ResNet for drowsiness classification
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        print("Loading models...")
        self.load_yolo()
        self.load_pose_detector()
        self.load_resnet()
        
        # State tracking
        self.drowsy_frame_count = 0
        self.distracted_frame_count = 0
        self.consecutive_frames_threshold = config['resnet']['consecutive_frames_drowsy']
        
        # Class names
        self.class_names = [config['data']['classes'][f'c{i}'] for i in range(10)]
        
        print("✓ All models loaded successfully!\n")
    
    def load_yolo(self):
        """Load YOLO model for object detection."""
        yolo_weights = 'models/yolo/weights/driver_detection/weights/best.pt'
        
        if os.path.exists(yolo_weights):
            self.yolo_model = YOLO(yolo_weights)
            print(f"  ✓ YOLO model loaded from {yolo_weights}")
        else:
            # Use pretrained model if custom weights not available
            print(f"  ⚠ Custom YOLO weights not found, using pretrained yolov8n")
            self.yolo_model = YOLO('yolov8n.pt')
    
    def load_pose_detector(self):
        """Load MediaPipe pose detector."""
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("  ✓ MediaPipe pose detector loaded")
    
    def load_resnet(self):
        """Load ResNet model for classification."""
        from torchvision import models
        
        checkpoint_path = 'models/resnet/checkpoints/best_model.pth'
        
        # Create model
        self.resnet_model = models.resnet50(pretrained=False)
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = torch.nn.Linear(num_features, 10)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.resnet_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ ResNet model loaded from {checkpoint_path}")
        else:
            print(f"  ⚠ ResNet checkpoint not found, using untrained model")
        
        self.resnet_model = self.resnet_model.to(self.device)
        self.resnet_model.eval()
        
        # Image preprocessing
        self.resnet_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_objects(self, frame):
        """
        Detect objects using YOLO.
        Returns bounding boxes for driver, face, hands, phone, etc.
        """
        results = self.yolo_model(frame, verbose=False)
        
        detections = {
            'driver': [],
            'face': [],
            'hands': [],
            'phone': [],
            'steering_wheel': []
        }
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class': cls
                }
                
                # Categorize detections (you may need to adjust class IDs)
                if cls == 0:  # driver
                    detections['driver'].append(detection)
                elif cls == 1:  # face
                    detections['face'].append(detection)
                elif cls == 2:  # hands
                    detections['hands'].append(detection)
                elif cls == 4:  # phone
                    detections['phone'].append(detection)
                elif cls == 3:  # steering wheel
                    detections['steering_wheel'].append(detection)
        
        return detections
    
    def extract_pose(self, frame):
        """Extract pose keypoints using MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        
        pose_data = {
            'landmarks': None,
            'head_angle': None,
            'eye_aspect_ratio': None,
            'posture': 'unknown'
        }
        
        if pose_results.pose_landmarks:
            pose_data['landmarks'] = pose_results.pose_landmarks
            pose_data['head_angle'] = self.calculate_head_angle(pose_results.pose_landmarks)
        
        if face_results.multi_face_landmarks:
            pose_data['eye_aspect_ratio'] = self.calculate_ear(
                face_results.multi_face_landmarks[0]
            )
        
        return pose_data, pose_results, face_results
    
    def calculate_head_angle(self, landmarks):
        """Calculate head tilt angle."""
        nose = landmarks.landmark[0]
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        
        shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_midpoint_y = (left_shoulder.y + right_shoulder.y) / 2
        
        diff_x = nose.x - shoulder_midpoint_x
        diff_y = nose.y - shoulder_midpoint_y
        
        angle = np.arctan2(diff_y, diff_x) * 180 / np.pi
        return angle
    
    def calculate_ear(self, face_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for drowsiness detection.
        Low EAR indicates closed eyes.
        """
        # MediaPipe face mesh eye landmarks
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        def eye_aspect_ratio(eye_landmarks):
            # Vertical distances
            A = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                              np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
            B = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - 
                              np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
            # Horizontal distance
            C = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                              np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
            
            ear = (A + B) / (2.0 * C)
            return ear
        
        left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
        right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
        
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        
        return (left_ear + right_ear) / 2.0
    
    def classify_driver_state(self, frame):
        """Classify driver state using ResNet."""
        # Preprocess frame
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = self.resnet_transform(frame_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.resnet_model(frame_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = int(predicted[0])
        confidence_score = float(confidence[0])
        class_name = self.class_names[predicted_class]
        
        return predicted_class, class_name, confidence_score
    
    def fuse_predictions(self, yolo_detections, pose_data, resnet_class):
        """
        Fuse predictions from all models to determine final alert.
        
        Returns:
            dict: {
                'status': 'alert' | 'drowsy' | 'distracted' | 'safe',
                'confidence': float,
                'reason': str
            }
        """
        weights = self.config['pipeline']['model_weights']
        
        # Initialize scores
        drowsy_score = 0.0
        distracted_score = 0.0
        
        # ResNet classification
        if resnet_class in [1, 2, 3, 4]:  # Texting/phone use
            distracted_score += weights['resnet'] * 1.0
        elif resnet_class in [6]:  # Drinking
            distracted_score += weights['resnet'] * 0.7
        elif resnet_class in [7, 8]:  # Reaching behind, hair/makeup
            distracted_score += weights['resnet'] * 0.8
        elif resnet_class in [9]:  # Talking to passenger
            distracted_score += weights['resnet'] * 0.5
        
        # YOLO detections
        if yolo_detections['phone']:
            distracted_score += weights['yolo'] * 1.0
        
        # Pose analysis
        if pose_data['eye_aspect_ratio'] is not None:
            ear_threshold = self.config['resnet']['eye_aspect_ratio_threshold']
            if pose_data['eye_aspect_ratio'] < ear_threshold:
                drowsy_score += weights['openpose'] * 1.0
        
        if pose_data['head_angle'] is not None:
            # Extreme head angles might indicate drowsiness
            if abs(pose_data['head_angle']) > 30:
                drowsy_score += weights['openpose'] * 0.5
        
        # Determine final status
        drowsy_threshold = self.config['pipeline']['drowsiness_threshold']
        distracted_threshold = self.config['pipeline']['distraction_threshold']
        
        if drowsy_score > drowsy_threshold:
            return {
                'status': 'drowsy',
                'confidence': drowsy_score,
                'reason': 'Eyes closed or head tilted'
            }
        elif distracted_score > distracted_threshold:
            return {
                'status': 'distracted',
                'confidence': distracted_score,
                'reason': 'Phone use or looking away'
            }
        else:
            return {
                'status': 'safe',
                'confidence': 1.0 - max(drowsy_score, distracted_score),
                'reason': 'Normal driving'
            }
    
    def process_frame(self, frame):
        """
        Process a single frame through the entire pipeline.
        
        Returns:
            Annotated frame and alert information
        """
        # 1. Object detection
        yolo_detections = self.detect_objects(frame)
        
        # 2. Pose estimation
        pose_data, pose_results, face_results = self.extract_pose(frame)
        
        # 3. Driver state classification
        resnet_class, class_name, confidence = self.classify_driver_state(frame)
        
        # 4. Fuse predictions
        alert = self.fuse_predictions(yolo_detections, pose_data, resnet_class)
        
        # Visualize results
        annotated_frame = self.visualize(frame, yolo_detections, pose_results,
                                         class_name, confidence, alert, pose_data)
        
        return annotated_frame, alert
    
    def visualize(self, frame, yolo_detections, pose_results, 
                 class_name, confidence, alert, pose_data):
        """Draw visualizations on frame."""
        annotated = frame.copy()
        
        # Draw YOLO bounding boxes
        for obj_type, detections in yolo_detections.items():
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{obj_type}: {det['confidence']:.2f}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        
        # Draw pose landmarks
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Draw status overlay
        status = alert['status']
        if status == 'drowsy':
            color = (0, 0, 255)  # Red
            text = "⚠ DROWSY ALERT!"
        elif status == 'distracted':
            color = (0, 165, 255)  # Orange
            text = "⚠ DISTRACTED!"
        else:
            color = (0, 255, 0)  # Green
            text = "✓ Safe Driving"
        
        # Status box
        cv2.rectangle(annotated, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (400, 150), color, 3)
        
        cv2.putText(annotated, text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(annotated, f"Class: {class_name}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Confidence: {confidence:.2f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Reason: {alert['reason']}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Additional info
        if pose_data['eye_aspect_ratio'] is not None:
            cv2.putText(annotated, f"EAR: {pose_data['eye_aspect_ratio']:.3f}",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if pose_data['head_angle'] is not None:
            cv2.putText(annotated, f"Head Angle: {pose_data['head_angle']:.1f}°",
                       (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated

def run_video_pipeline(video_path, config, output_path=None):
    """Run pipeline on a video file."""
    # Initialize system
    system = DriverMonitoringSystem(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}\n")
    
    # Video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    alerts_count = {'safe': 0, 'drowsy': 0, 'distracted': 0}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, alert = system.process_frame(frame)
        
        # Track alerts
        alerts_count[alert['status']] += 1
        
        # Save output
        if output_path:
            out.write(annotated_frame)
        
        # Display
        cv2.imshow('Driver Monitoring System', annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Safe frames: {alerts_count['safe']} ({100*alerts_count['safe']/frame_count:.1f}%)")
    print(f"Drowsy frames: {alerts_count['drowsy']} ({100*alerts_count['drowsy']/frame_count:.1f}%)")
    print(f"Distracted frames: {alerts_count['distracted']} ({100*alerts_count['distracted']/frame_count:.1f}%)")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Driver Monitoring System Pipeline')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID for live feed')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'camera'],
                       help='Mode: process video file or live camera feed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    print("=" * 60)
    print("Driver Monitoring System - Inference Pipeline")
    print("=" * 60)
    
    if args.mode == 'video':
        if not args.video:
            print("Error: --video argument required for video mode")
            return
        run_video_pipeline(args.video, config, args.output)
    else:
        # Live camera feed
        run_video_pipeline(args.camera, config, args.output)

if __name__ == "__main__":
    main()
