# Driver Distraction & Drowsiness Detection System

A collaborative project by **Srividya Ravi Sivashankar**, **Sonia Poh**, and **Lynus Phua** that uses advanced Computer Vision techniques to detect driver distraction and drowsiness in real-time, enhancing road safety.

## 🎯 Project Overview

This system combines **four state-of-the-art models** to create a comprehensive driver monitoring solution:

1. **Grounding DINO** - Auto-annotates images with bounding boxes for objects (driver, face, hands, phone, etc.)
2. **YOLOv8** - Real-time object detection for driver and distraction-related objects
3. **MediaPipe/OpenPose** - Pose estimation for head position, body posture, and drowsiness indicators
4. **ResNet** - Deep learning classification for driver state (safe, drowsy, distracted)

## 📊 Dataset

The project uses the **State Farm Driver Distraction Dataset** with 10 classes:
- **c0**: Safe driving
- **c1**: Texting - right hand
- **c2**: Talking on phone - right hand
- **c3**: Texting - left hand
- **c4**: Talking on phone - left hand
- **c5**: Operating radio
- **c6**: Drinking
- **c7**: Reaching behind
- **c8**: Hair and makeup
- **c9**: Talking to passenger

Total: **~3,265 images** organized by class

## 🏗️ Project Structure

```
DAP-Project/
├── config/
│   └── config.yaml              # Centralized configuration
├── data/
│   ├── raw/                     # Original training data
│   ├── splits/                  # Train/val/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── annotated/               # Grounding DINO outputs
│   └── preprocessed/            # Extracted faces and poses
├── models/
│   ├── grounding_dino/          # Auto-annotation model
│   ├── yolo/                    # Object detection
│   │   ├── config/
│   │   └── weights/
│   ├── openpose/                # Pose estimation
│   └── resnet/                  # Drowsiness classification
│       ├── weights/
│       └── checkpoints/
├── scripts/
│   ├── 00_split_data.py         # Split dataset
│   ├── 01_annotate_grounding_dino.py  # Auto-annotation
│   ├── 02_prepare_yolo_data.py  # YOLO data prep
│   ├── 03_train_yolo.py         # Train YOLO
│   ├── 04_extract_poses.py      # Extract pose keypoints
│   ├── 05_train_resnet.py       # Train ResNet classifier
│   └── pipeline.py              # Full inference pipeline
├── utils/
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── visualization.py         # Visualization tools
│   └── metrics.py               # Evaluation metrics
├── notebooks/                   # Jupyter notebooks for analysis
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DAP-Project-
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Grounding DINO** (requires manual installation)
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

5. **Download pre-trained weights**
- Grounding DINO: Download from [official repo](https://github.com/IDEA-Research/GroundingDINO)
- YOLOv8: Auto-downloads on first run
- ResNet: Uses PyTorch pretrained weights

## 📋 Step-by-Step Workflow

### Phase 1: Data Preparation

**Step 1: Split the dataset**
```bash
python scripts/00_split_data.py
```
This splits your data into:
- 70% training
- 15% validation
- 15% testing

**Output**: `data/splits/{train,val,test}/c0-c9/`

---

### Phase 2: Auto-Annotation with Grounding DINO

**Step 2: Auto-annotate images**
```bash
python scripts/01_annotate_grounding_dino.py
```
Grounding DINO detects:
- Driver location
- Face region
- Hands
- Steering wheel
- Phone/objects

**Output**: YOLO-format annotations in `data/annotated/`

---

### Phase 3: YOLO Object Detection

**Step 3: Prepare YOLO dataset**
```bash
python scripts/02_prepare_yolo_data.py
```
Creates YOLO dataset configuration file.

**Step 4: Train YOLO model**
```bash
python scripts/03_train_yolo.py
```
Trains YOLOv8 for 100 epochs (customizable in `config.yaml`)

**Output**: Best model at `models/yolo/weights/driver_detection/weights/best.pt`

---

### Phase 4: Pose Extraction

**Step 5: Extract pose keypoints**
```bash
python scripts/04_extract_poses.py
```
Uses MediaPipe to extract:
- Body pose (25 keypoints)
- Face landmarks (468 points)
- Head angle
- Eye aspect ratio (EAR)

**Output**: JSON files in `data/preprocessed/poses/`

---

### Phase 5: ResNet Classification

**Step 6: Train ResNet classifier**
```bash
python scripts/05_train_resnet.py
```
Fine-tunes ResNet50 on the 10 distraction classes.

**Output**: 
- Best model: `models/resnet/checkpoints/best_model.pth`
- Confusion matrix and metrics

---

### Phase 6: Full Pipeline Inference

**Step 7: Run the complete pipeline**

For **video file**:
```bash
python scripts/pipeline.py --mode video --video path/to/video.mp4 --output results/output.mp4
```

For **live camera**:
```bash
python scripts/pipeline.py --mode camera --camera 0
```

The pipeline:
1. Detects objects with YOLO
2. Extracts pose with MediaPipe
3. Classifies driver state with ResNet
4. Fuses predictions to generate alerts
5. Displays real-time visualization

**Press 'q' to quit**

---

## ⚙️ Configuration

All settings are in [`config/config.yaml`](config/config.yaml). Key parameters:

```yaml
# Data splits
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# YOLO training
yolo:
  model: "yolov8n.pt"
  epochs: 100
  batch_size: 16

# ResNet training
resnet:
  architecture: "resnet50"
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

# Alert thresholds
pipeline:
  drowsiness_threshold: 0.6
  distraction_threshold: 0.7
```

## 📊 Model Fusion Strategy

The system uses **weighted average fusion** to combine predictions:

```
Final Score = 0.3 × YOLO + 0.3 × OpenPose + 0.4 × ResNet
```

**Alert Triggers**:
- **Drowsy**: EAR < 0.25 for 20+ frames OR ResNet detects drowsiness
- **Distracted**: Phone detected OR looking away OR unsafe hand position

## 🎯 Performance Metrics

After training, evaluate using:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class metrics
- Mean Average Precision (mAP) for YOLO

## 📝 Data Flow Summary

```
Raw Images (c0-c9)
    ↓
[Split] → Train/Val/Test
    ↓
[Grounding DINO] → Auto-annotated with bboxes
    ↓
[YOLO Training] → Object detector for driver/face/hands/phone
    ↓
[MediaPipe] → Pose keypoints + Head angle + EAR
    ↓
[ResNet Training] → Classify 10 distraction states
    ↓
[Pipeline] → Fuse all models → Real-time alerts
```

## 🔧 Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in `config.yaml`

**Issue**: Grounding DINO not installed
- **Solution**: Follow manual installation instructions above

**Issue**: Low accuracy
- **Solution**: 
  - Increase training epochs
  - Adjust learning rates
  - Add more data augmentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

[Add your license here]

## 👥 Team

- **Srividya Ravi Sivashankar**
- **Sonia Poh**
- **Lynus Phua**

## 🙏 Acknowledgments

- State Farm for the distraction dataset
- Grounding DINO team
- Ultralytics for YOLOv8
- MediaPipe team
- PyTorch community

---

**For questions or issues, please open a GitHub issue or contact the team.**
