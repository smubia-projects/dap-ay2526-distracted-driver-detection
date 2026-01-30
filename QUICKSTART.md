# Quick Start Guide

## 🎯 What You Need to Do

This guide walks you through the **complete workflow** for your Driver Distraction Detection project.

---

## 📦 Installation (One-Time Setup)

### 1. Install Python Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Install Grounding DINO (Manual)
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### 3. Download Grounding DINO Weights
```bash
# Create directory
mkdir -p models/grounding_dino

# Download weights (example)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O models/grounding_dino/groundingdino_swint_ogc.pth
```

---

## 🔄 Training Workflow (Follow in Order)

### **Phase 1: Data Preparation** ✅

**Current Status**: You have `training_data_with_all_classes/train/c0-c9/` with ~3,265 images

**Action**: Split data into train/val/test sets
```bash
python scripts/00_split_data.py
```

**What it does**:
- Splits images: 70% train, 15% val, 15% test
- Maintains class balance (stratified split)
- Outputs to `data/splits/`

**Expected Output**:
```
data/splits/
├── train/      (~2,285 images)
├── val/        (~490 images)
└── test/       (~490 images)
```

---

### **Phase 2: Auto-Annotation** 🏷️

**Action**: Annotate images with bounding boxes using Grounding DINO
```bash
python scripts/01_annotate_grounding_dino.py
```

**What it does**:
- Detects: driver, face, hands, steering wheel, phone, cup
- Generates YOLO-format annotations (.txt files)
- Saves to `data/annotated/`

**Time**: ~15-30 minutes for 3,265 images (GPU recommended)

---

### **Phase 3: YOLO Training** 🎯

**Step 3a**: Prepare YOLO dataset
```bash
python scripts/02_prepare_yolo_data.py
```
Creates `models/yolo/config/dataset.yaml`

**Step 3b**: Train YOLO model
```bash
python scripts/03_train_yolo.py
```

**What it does**:
- Trains YOLOv8 for object detection
- 100 epochs (~2-4 hours on GPU)
- Saves best model to `models/yolo/weights/`

**Monitor Training**: Results saved in `models/yolo/weights/driver_detection/`

---

### **Phase 4: Pose Extraction** 🧍

**Action**: Extract pose keypoints from all images
```bash
python scripts/04_extract_poses.py
```

**What it does**:
- Uses MediaPipe to detect 25 body keypoints + 468 face landmarks
- Calculates head angle and Eye Aspect Ratio (EAR)
- Saves JSON files to `data/preprocessed/poses/`

**Time**: ~10-20 minutes

---

### **Phase 5: ResNet Training** 🧠

**Action**: Train ResNet classifier for driver state
```bash
python scripts/05_train_resnet.py
```

**What it does**:
- Fine-tunes ResNet50 on 10 distraction classes
- 50 epochs (~1-2 hours on GPU)
- Saves best model to `models/resnet/checkpoints/best_model.pth`

**Outputs**:
- Confusion matrix: `models/resnet/checkpoints/confusion_matrix.png`
- Classification report with accuracy, precision, recall

---

### **Phase 6: Run Full Pipeline** 🚀

**Action**: Test the complete system on video or camera

**For testing on video**:
```bash
python scripts/pipeline.py --mode video --video test_video.mp4 --output results/output.mp4
```

**For live camera feed**:
```bash
python scripts/pipeline.py --mode camera --camera 0
```

**What it does**:
- Runs YOLO + MediaPipe + ResNet in real-time
- Fuses predictions to detect drowsiness/distraction
- Shows live visualization with alerts

**Press 'q' to quit**

---

## 📊 Expected Timeline

| Phase | Task | Time | GPU Required |
|-------|------|------|--------------|
| 1 | Data splitting | 1 min | No |
| 2 | Auto-annotation | 15-30 min | Yes (recommended) |
| 3 | YOLO training | 2-4 hours | Yes |
| 4 | Pose extraction | 10-20 min | No |
| 5 | ResNet training | 1-2 hours | Yes |
| 6 | Testing pipeline | Minutes | Yes (for real-time) |

**Total**: ~4-7 hours (mostly automated training)

---

## 🎛️ Customization

### Adjust Training Parameters

Edit [`config/config.yaml`](config/config.yaml):

```yaml
# Speed up training (lower quality)
yolo:
  epochs: 50  # Reduce from 100
  batch_size: 32  # Increase if you have GPU memory

resnet:
  epochs: 30  # Reduce from 50
  batch_size: 64

# Adjust alert sensitivity
pipeline:
  drowsiness_threshold: 0.5  # Lower = more sensitive
  distraction_threshold: 0.6
```

---

## 🐛 Common Issues

### **Issue 1**: Out of GPU memory
```bash
# Solution: Reduce batch size in config.yaml
yolo:
  batch_size: 8  # Reduce from 16
resnet:
  batch_size: 16  # Reduce from 32
```

### **Issue 2**: Grounding DINO not working
```bash
# Check installation
python -c "import groundingdino; print('OK')"

# If fails, reinstall
cd GroundingDINO && pip install -e . --force-reinstall
```

### **Issue 3**: Slow training
- Use GPU: Check `torch.cuda.is_available()` returns `True`
- Reduce image size in `config.yaml`: `image_size: [416, 416]`

---

## ✅ Validation Checkpoints

After each phase, verify:

**After Phase 1**:
```bash
ls data/splits/train/c0/  # Should show images
```

**After Phase 2**:
```bash
ls data/annotated/train/labels/c0/  # Should show .txt annotation files
```

**After Phase 3**:
```bash
ls models/yolo/weights/driver_detection/weights/best.pt  # Should exist
```

**After Phase 5**:
```bash
ls models/resnet/checkpoints/best_model.pth  # Should exist
```

---

## 📈 Next Steps After Training

1. **Evaluate Models**: Check confusion matrices and metrics
2. **Fine-tune Hyperparameters**: Adjust based on validation results
3. **Deploy**: Integrate pipeline into a real vehicle system
4. **Extend**: Add more classes or improve detection accuracy

---

## 🆘 Need Help?

1. Check the detailed [README.md](README.md)
2. Review `config/config.yaml` for all parameters
3. Contact team members
4. Open a GitHub issue

**Good luck with your project! 🚗💨**
