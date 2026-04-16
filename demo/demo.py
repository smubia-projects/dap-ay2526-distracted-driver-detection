"""
Multi-Model Driver Distraction Detection
=========================================
Tab 1: Classification   — Original model selection + prediction
Tab 2: Grad-CAM         — Heatmap overlays showing where the model looks
Tab 3: LIME             — Superpixel-level explanation of predictions
"""

import gradio as gr
import torch
import torch.nn.functional as Fn
import numpy as np
from torchvision import transforms, models as tv_models
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from gradcam import GradCAMExtractor, AttentionRolloutExtractor, vit_reshape_transform, overlay_heatmap

# ==================== Configuration ====================

VIT_PATH = Path("vit_auc_cam1.pth")
RESNET_PATH = Path("resnet50_auc_cam1.pth")
EFFICIENTNET_PATH = Path("efficientnet_auc_cam1.pth")
STACKING_PATH = Path("stacking_meta_learner.pth")
BLENDING_PATH = Path("blending_weights.pth")
DEVICE = torch.device('cpu')

CLASS_NAMES = {
    'c0': 'Safe Driving',
    'c1': 'Texting - Right',
    'c2': 'Talking on Phone - Right',
    'c3': 'Texting - Left',
    'c4': 'Talking on Phone - Left',
    'c5': 'Operating Radio',
    'c6': 'Drinking',
    'c7': 'Reaching Behind',
    'c8': 'Hair and Makeup',
    'c9': 'Talking to Passenger'
}

CLASS_LIST = [f"c{i}: {CLASS_NAMES[f'c{i}']}" for i in range(10)]
NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== Load Models ====================

print("=" * 70)
print("Loading Models...")
print("=" * 70)

print("\n1. Loading Vision Transformer...")
try:
    import timm
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    vit_model.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE, weights_only=True))
    vit_model.eval().to(DEVICE)
    print("   ✓ Vision Transformer loaded")
    has_vit = True
except Exception as e:
    print(f"   ✗ Vision Transformer failed: {e}")
    has_vit = False

print("\n2. Loading ResNet50...")
try:
    resnet_model = tv_models.resnet50(weights=None)
    resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
    resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE, weights_only=True))
    resnet_model.eval().to(DEVICE)
    print("   ✓ ResNet50 loaded")
    has_resnet = True
except Exception as e:
    print(f"   ✗ ResNet50 failed: {e}")
    has_resnet = False

print("\n3. Loading EfficientNet...")
try:
    efficientnet_model = tv_models.efficientnet_b0(weights=None)
    efficientnet_model.classifier[1] = torch.nn.Linear(efficientnet_model.classifier[1].in_features, NUM_CLASSES)
    efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=DEVICE, weights_only=True))
    efficientnet_model.eval().to(DEVICE)
    print("   ✓ EfficientNet loaded")
    has_efficientnet = True
except Exception as e:
    print(f"   ✗ EfficientNet failed: {e}")
    has_efficientnet = False

print("\n4. Loading Stacking Meta Learner...")
try:
    stacking_model = torch.load(STACKING_PATH, map_location=DEVICE, weights_only=False)
    print("   ✓ Stacking Meta Learner loaded")
    has_stacking = True
except Exception as e:
    print(f"   ✗ Stacking Meta Learner failed: {e}")
    has_stacking = False

print("\n3. Loading Blending Weights...")
try:
    blending_weights = torch.load(BLENDING_PATH, map_location=DEVICE, weights_only=False)
    print(f"   ✓ Blending weights loaded: {blending_weights}")
    has_blending = True
except Exception as e:
    print(f"   ✗ Blending weights failed: {e}")
    has_blending = False

print("\n" + "=" * 70)
print(f"ViT: {has_vit} | ResNet: {has_resnet} | EfficientNet: {has_efficientnet} | Stacking: {has_stacking} | Blending: {has_blending}")
print("=" * 70 + "\n")

# Instantiate once — hooks register on construction; re-creating every call
# would accumulate duplicate hooks on the model.
# GradCAM targets blocks[-1].norm1 (NOT blocks[-1] — patch gradients are zero
# at the block output because only the CLS token feeds the classifier).
vit_gradcam   = GradCAMExtractor(vit_model, vit_model.blocks[-1].norm1,
                                  reshape_transform=vit_reshape_transform) if has_vit else None
vit_rollout   = AttentionRolloutExtractor(vit_model) if has_vit else None

# ==================== Classification Helpers ====================

def get_base_probabilities(img_tensor):
    """Get base model probabilities (used as input to stacking/blending)."""
    probs = {}
    with torch.no_grad():
        if has_vit:
            probs['ViT'] = torch.softmax(vit_model(img_tensor), dim=1).cpu().numpy()
        if has_resnet:
            probs['ResNet'] = torch.softmax(resnet_model(img_tensor), dim=1).cpu().numpy()
        if has_efficientnet:
            probs['EfficientNet'] = torch.softmax(efficientnet_model(img_tensor), dim=1).cpu().numpy()
    return probs

def predict_vit(img_tensor):
    with torch.no_grad():
        return torch.softmax(vit_model(img_tensor), dim=1)[0]

def predict_stacking(img_tensor):
    probs = get_base_probabilities(img_tensor)
    if not probs:
        raise RuntimeError("No base models available for stacking")
    meta_input = np.concatenate(list(probs.values()), axis=1)
    pred_probs = stacking_model.predict_proba(meta_input)[0]
    return torch.tensor(pred_probs, dtype=torch.float32)

def predict_blending(img_tensor):
    probs = get_base_probabilities(img_tensor)
    total_weight = 0.0
    blended = None
    for model_name, weight in blending_weights.items():
        if model_name in probs:
            p = torch.tensor(probs[model_name], dtype=torch.float32)[0]
            blended = p * weight if blended is None else blended + p * weight
            total_weight += weight
    if blended is None:
        raise RuntimeError("No base model predictions available for blending")
    return blended / total_weight

def predict_image(image, model_selection):
    if image is None:
        return {"No image provided": 1.0}
    try:
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        if model_selection == "Vision Transformer":
            if not has_vit:
                return {"Model not loaded": 1.0}
            probabilities = predict_vit(img_tensor)
        elif model_selection == "Stacking Meta Learner":
            if not has_stacking:
                return {"Model not loaded": 1.0}
            probabilities = predict_stacking(img_tensor)
        elif model_selection == "Blending Ensemble":
            if not has_blending:
                return {"Model not loaded": 1.0}
            probabilities = predict_blending(img_tensor)
        else:
            return {"Invalid model selection": 1.0}
    except Exception as e:
        return {f"Error: {str(e)[:80]}": 1.0}

    results = {}
    for class_code, class_name in CLASS_NAMES.items():
        idx = int(class_code[1])
        results[f"{class_code}: {class_name}"] = probabilities[idx].item()
    return results


# ==================== Grad-CAM Helpers ====================

def _format_prediction(probs_np, pred_idx):
    """Format top-3 predictions as readable text."""
    top3 = np.argsort(probs_np)[::-1][:3]
    lines = [f"Predicted: {CLASS_LIST[pred_idx]}  ({probs_np[pred_idx]*100:.1f}%)\n"]
    lines.append("Top-3 predictions:")
    for rank, idx in enumerate(top3, 1):
        lines.append(f"  {rank}. {CLASS_LIST[idx]} — {probs_np[idx]*100:.1f}%")
    return "\n".join(lines)


def run_gradcam(image, model_name, method, alpha):
    if image is None:
        return Image.new("RGB", (224, 224)), "Please upload an image."
    if not has_vit:
        return Image.new("RGB", (224, 224)), "ViT model is not loaded."

    try:
        img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        # Generate heatmap
        if method == "Grad-CAM":
            heatmap, _ = vit_gradcam.generate(img_tensor)
        else:  # Attention Rollout
            heatmap, _ = vit_rollout.generate(img_tensor)

        # Get probabilities for the selected model
        if model_name == "ViT":
            with torch.no_grad():
                probs = torch.softmax(vit_model(img_tensor), dim=1)[0].cpu().numpy()
        elif model_name == "Stacking":
            if not has_stacking:
                return Image.new("RGB", (224, 224)), "Stacking model is not loaded."
            probs = predict_stacking(img_tensor).numpy()
        elif model_name == "Blending":
            if not has_blending:
                return Image.new("RGB", (224, 224)), "Blending weights are not loaded."
            probs = predict_blending(img_tensor).numpy()
        else:
            return Image.new("RGB", (224, 224)), f"Unknown model: {model_name}"

        pred_idx = int(np.argmax(probs))
        overlay  = overlay_heatmap(image.convert("RGB"), heatmap, alpha=alpha)
        text     = _format_prediction(probs, pred_idx)
        return overlay, text

    except Exception as e:
        return Image.new("RGB", (224, 224)), f"Error: {str(e)}"


def run_gradcam_compare(image, method, alpha):
    if image is None:
        blank = Image.new("RGB", (224, 224))
        msg   = "Please upload an image."
        return blank, msg, blank, msg, blank, msg

    results = []
    for name in ["ViT", "Stacking", "Blending"]:
        try:
            img, txt = run_gradcam(image, name, method, alpha)
            results.extend([img, txt])
        except Exception as e:
            results.extend([Image.new("RGB", (224, 224)), f"Error: {e}"])
    return tuple(results)


# ==================== LIME Helpers ====================

def _make_lime_predict_fn(model_name):
    """Return a LIME-compatible predict function (batch numpy images → probabilities)."""
    def _batch_to_tensors(images):
        tensors = [transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0) for img in images]
        return torch.cat(tensors).to(DEVICE)

    if model_name == "ViT":
        def predict_fn(images):
            with torch.no_grad():
                return torch.softmax(vit_model(_batch_to_tensors(images)), dim=1).cpu().numpy()
    elif model_name == "ResNet":
        def predict_fn(images):
            with torch.no_grad():
                return torch.softmax(resnet_model(_batch_to_tensors(images)), dim=1).cpu().numpy()
    elif model_name == "EfficientNet":
        def predict_fn(images):
            with torch.no_grad():
                return torch.softmax(efficientnet_model(_batch_to_tensors(images)), dim=1).cpu().numpy()
    elif model_name == "Stacking":
        def predict_fn(images):
            results = []
            for img in images:
                t = transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(DEVICE)
                results.append(predict_stacking(t).numpy())
            return np.array(results)
    elif model_name == "Blending":
        def predict_fn(images):
            results = []
            for img in images:
                t = transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(DEVICE)
                results.append(predict_blending(t).numpy())
            return np.array(results)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return predict_fn


def run_lime_compare(image, num_samples, num_features):
    """Run LIME on ViT, Stacking, and Blending and return side-by-side results."""
    if image is None:
        blank = Image.new("RGB", (224, 224))
        msg = "Please upload an image."
        return blank, msg, blank, msg, blank, msg

    model_available = {
        "ViT": has_vit,
        "Stacking": has_stacking, "Blending": has_blending,
    }

    results = []
    for model_name in ["ViT", "Stacking", "Blending"]:
        if not model_available.get(model_name, False):
            results.extend([Image.new("RGB", (224, 224)), f"{model_name} model is not loaded."])
            continue
        try:
            img, txt = run_lime(image, model_name, num_samples, num_features)
            results.extend([img, txt])
        except Exception as e:
            results.extend([Image.new("RGB", (224, 224)), f"Error: {e}"])

    return tuple(results)


def run_lime(image, model_name, num_samples, num_features):
    if image is None:
        return Image.new("RGB", (224, 224)), "Please upload an image."

    model_available = {
        "ViT": has_vit, "ResNet": has_resnet, "EfficientNet": has_efficientnet,
        "Stacking": has_stacking, "Blending": has_blending,
    }
    if not model_available.get(model_name, False):
        return Image.new("RGB", (224, 224)), f"{model_name} model is not loaded."

    try:
        from lime import lime_image as lime_img_module

        img_rgb = image.convert("RGB").resize((224, 224))
        img_np = np.array(img_rgb)

        predict_fn = _make_lime_predict_fn(model_name)

        explainer = lime_img_module.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=1,
            hide_color=None,
            num_samples=int(num_samples),
            random_seed=42,
        )

        pred_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            pred_label,
            positive_only=False,
            num_features=int(num_features),
            hide_rest=False,
        )

        # Build green/red overlay: green = supports prediction, red = contradicts
        fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
        ax.imshow(mark_boundaries(temp / 255.0, mask))
        ax.axis('off')
        ax.set_title(f"{model_name} — LIME Explanation", fontsize=11, pad=6)
        fig.tight_layout(pad=0.5)

        # Convert matplotlib figure to PIL image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        result_img = Image.open(buf).convert("RGB")

        # Format prediction text
        probs_np = predict_fn(np.array([img_np]))[0]
        text = _format_prediction(probs_np, pred_label)
        return result_img, text

    except ImportError:
        return Image.new("RGB", (224, 224)), "LIME not installed. Run: pip install lime"
    except Exception as e:
        return Image.new("RGB", (224, 224)), f"Error: {str(e)}"


# ==================== Gradio Interface (Blocks with Tabs) ====================

blending_vit_w = blending_weights.get('ViT', 0) if has_blending else 0

with gr.Blocks(title="Driver Distraction Detection") as demo:

    gr.Markdown("# Driver Distraction Detection — Multi-Model System")
    gr.Markdown(
        "Detect driver distraction using your choice of advanced models, "
        "Visualise model attention with Grad-CAM and explain predictions with LIME."
    )

    # ──────────────────────────────────────────
    #  TAB 1 — CLASSIFICATION
    # ──────────────────────────────────────────
    with gr.Tab("Classification"):
        gr.Markdown(f"""
### Select a model and upload a driver image

**Available Models:**
- **Vision Transformer (ViT):** ViT-Base fine-tuned on AUC dataset
- **Stacking Meta Learner:** LogisticRegression meta-learner on top of ViT predictions
- **Blending Ensemble:** Weighted average of ViT predictions (ViT weight: {blending_vit_w:.2f})
        """)

        with gr.Row():
            cls_image = gr.Image(type="pil", label="Upload Driver Image")
            with gr.Column():
                cls_model = gr.Dropdown(
                    choices=["Vision Transformer", "Stacking Meta Learner", "Blending Ensemble"],
                    value="Vision Transformer",
                    label="Select Model",
                )
                cls_btn = gr.Button("Classify", variant="primary")

        cls_output = gr.Label(num_top_classes=10, label="Distraction Prediction")
        cls_btn.click(fn=predict_image, inputs=[cls_image, cls_model], outputs=cls_output)

        gr.Markdown("""
---
### Classification Categories
| Code | Behaviour |
|------|-----------|
| c0 | Safe Driving |
| c1 | Texting - Right |
| c2 | Talking on Phone - Right |
| c3 | Texting - Left |
| c4 | Talking on Phone - Left |
| c5 | Operating Radio |
| c6 | Drinking |
| c7 | Reaching Behind |
| c8 | Hair and Makeup |
| c9 | Talking to Passenger |
        """)

    # ──────────────────────────────────────────
    #  TAB 2 — GRAD-CAM
    # ──────────────────────────────────────────
    with gr.Tab("Grad-CAM Visualisation"):
        gr.Markdown("""
## Where is the model looking?

Upload a driver image to see a heatmap overlay of the regions driving the prediction.
**Warm colours = high activation. Cool colours = low activation.**

- **Grad-CAM** — class-discriminative; hooks `blocks[-1].norm1` with a reshape transform
- **Attention Rollout** — spatially coherent; accumulates attention across all layers with identity residual
        """)

        with gr.Row():
            gc_image = gr.Image(type="pil", label="Upload Driver Image")
            with gr.Column():
                gc_method = gr.Radio(
                    choices=["Grad-CAM", "Attention Rollout"],
                    value="Grad-CAM",
                    label="Visualisation Method",
                )
                gc_alpha = gr.Slider(
                    minimum=0.2, maximum=0.8, value=0.5, step=0.05,
                    label="Heatmap Opacity",
                )

        with gr.Tab("Single Model"):
            gc_model = gr.Dropdown(
                choices=["ViT", "Stacking", "Blending"],
                value="ViT",
                label="Select Model",
            )
            gc_btn = gr.Button("Generate Heatmap", variant="primary")

            with gr.Row():
                gc_overlay = gr.Image(type="pil", label="Heatmap Overlay")
                gc_text    = gr.Textbox(label="Prediction", lines=6)

            gc_btn.click(
                fn=run_gradcam,
                inputs=[gc_image, gc_model, gc_method, gc_alpha],
                outputs=[gc_overlay, gc_text],
            )

        with gr.Tab("Compare All Models"):
            gc_compare_btn = gr.Button("Run All Models", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ViT")
                    gc_vit_img = gr.Image(type="pil", label="ViT")
                    gc_vit_txt = gr.Textbox(label="ViT Prediction", lines=5)
                with gr.Column():
                    gr.Markdown("### Stacking")
                    gc_stk_img = gr.Image(type="pil", label="Stacking")
                    gc_stk_txt = gr.Textbox(label="Stacking Prediction", lines=5)
                with gr.Column():
                    gr.Markdown("### Blending")
                    gc_bld_img = gr.Image(type="pil", label="Blending")
                    gc_bld_txt = gr.Textbox(label="Blending Prediction", lines=5)

            gc_compare_btn.click(
                fn=run_gradcam_compare,
                inputs=[gc_image, gc_method, gc_alpha],
                outputs=[gc_vit_img, gc_vit_txt, gc_stk_img, gc_stk_txt, gc_bld_img, gc_bld_txt],
            )

    # ──────────────────────────────────────────
    #  TAB 3 — LIME
    # ──────────────────────────────────────────
    with gr.Tab("LIME Explainability"):
        gr.Markdown("""
## Which image regions drive the prediction?

**LIME** (Local Interpretable Model-agnostic Explanations) perturbs superpixels
of the image to measure each region's contribution to the predicted class.

- **Green boundaries** = regions that *support* the prediction
- **Red boundaries** = regions that *contradict* the prediction

> Note: higher sample counts give more stable explanations but take longer to run.
        """)

        with gr.Row():
            lime_image_input = gr.Image(type="pil", label="Upload Driver Image")
            with gr.Column():
                lime_num_samples = gr.Slider(
                    minimum=100, maximum=1000, value=300, step=50,
                    label="LIME Samples (higher = more stable, slower)",
                )
                lime_num_features = gr.Slider(
                    minimum=3, maximum=20, value=10, step=1,
                    label="Number of Superpixel Features to Highlight",
                )

        # ── Single model ──
        with gr.Tab("Single Model"):
            lime_model = gr.Dropdown(
                choices=["ViT", "Stacking", "Blending"],
                value="ViT",
                label="Select Model",
            )
            lime_btn = gr.Button("Run LIME", variant="primary")

            with gr.Row():
                lime_output_img = gr.Image(type="pil", label="LIME Explanation")
                lime_output_txt = gr.Textbox(label="Prediction", lines=6)

            lime_btn.click(
                fn=run_lime,
                inputs=[lime_image_input, lime_model, lime_num_samples, lime_num_features],
                outputs=[lime_output_img, lime_output_txt],
            )

        # ── Compare all models ──
        with gr.Tab("Compare All Models"):
            lime_compare_btn = gr.Button("Run All Models", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ViT")
                    lime_vit_img = gr.Image(type="pil", label="ViT")
                    lime_vit_txt = gr.Textbox(label="ViT Prediction", lines=5)
                with gr.Column():
                    gr.Markdown("### Stacking Meta Learner")
                    lime_stack_img = gr.Image(type="pil", label="Stacking")
                    lime_stack_txt = gr.Textbox(label="Stacking Prediction", lines=5)
                with gr.Column():
                    gr.Markdown("### Blending Ensemble")
                    lime_blend_img = gr.Image(type="pil", label="Blending")
                    lime_blend_txt = gr.Textbox(label="Blending Prediction", lines=5)

            lime_compare_btn.click(
                fn=run_lime_compare,
                inputs=[lime_image_input, lime_num_samples, lime_num_features],
                outputs=[
                    lime_vit_img, lime_vit_txt,
                    lime_stack_img, lime_stack_txt,
                    lime_blend_img, lime_blend_txt,
                ],
            )

    gr.Markdown(f"---\n**Dataset:** AUC Distracted Driver Dataset v2 — Camera 1 &nbsp;|&nbsp; **Device:** {DEVICE}")


# ==================== Launch ====================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LAUNCHING GRADIO — CLASSIFICATION + GRAD-CAM")
    print("=" * 70)
    print(f"ViT: {has_vit} | ResNet: {has_resnet} | EfficientNet: {has_efficientnet} | Stacking: {has_stacking} | Blending: {has_blending}")
    print("=" * 70 + "\n")

    demo.launch(share=True)
