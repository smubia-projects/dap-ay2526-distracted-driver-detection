---
title: Distracted Driver Detection
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.9.1"
app_file: demo.py
pinned: false
---

# Driver Distraction Detection — Multi-Model System

Classify driver distraction from a single image using an ensemble of Vision Transformer, ResNet50, and EfficientNet models trained on the AUC Distracted Driver Dataset.

**Three tabs:**
- **Classification** — upload a driver image, get distraction class probabilities from ViT, Stacking, or Blending ensemble
- **Grad-CAM** — heatmap overlay showing which regions the model attends to
- **LIME** — superpixel-level explanation of the prediction

**10 distraction classes:** safe driving · texting (left/right) · phone call (left/right) · operating radio · drinking · reaching behind · hair/makeup · talking to passenger

---

*Project by Srividya Ravi Sivashankar, Sonia Poh, and Lynus Phua — NUS BIA Data Associate Programme AY25/26*
