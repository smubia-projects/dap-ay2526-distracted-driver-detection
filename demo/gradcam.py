"""
Grad-CAM & Attention Rollout for Driver Distraction Detection
=============================================================
References:
  - Selvaraju et al. 2017  — Grad-CAM
  - Abnar & Zuidema 2020   — Attention Rollout
  - jacobgil/pytorch-grad-cam ViT tutorial
  - jacobgil/vit-explain

GradCAMExtractor:
    Works for CNNs (B,C,H,W feature maps) and timm ViT (B,N,C patch tokens).
    For ViT pass reshape_transform=vit_reshape_transform and target
    model.blocks[-1].norm1  — the last block *output* has zero patch gradients
    because classification only reads the CLS token.

AttentionRolloutExtractor:
    Class-agnostic but spatially coherent. Hooks attn_drop in every block
    (requires fused_attn=False), accumulates attention with identity residual.

overlay_heatmap:
    Blends any (H,W) float heatmap onto a PIL image.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


# ─────────────────────────────────────────────
#  RESHAPE TRANSFORM  (ViT only)
# ─────────────────────────────────────────────

def vit_reshape_transform(tensor, height=14, width=14):
    """
    Convert ViT patch-token tensor to CNN-style spatial feature map.

    Input : (B, N, C)  where N = 1 cls + height*width patch tokens
    Output: (B, C, height, width)
    """
    result = tensor[:, 1:, :]                                          # drop CLS
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)                    # B,C,H,W
    return result


# ─────────────────────────────────────────────
#  GRAD-CAM  (CNN + ViT)
# ─────────────────────────────────────────────

class GradCAMExtractor:
    """
    Standard Grad-CAM that supports both CNNs and timm ViT.

    Parameters
    ----------
    model            : torch.nn.Module
    target_layer     : torch.nn.Module
        CNN  →  model.layer4[-1]          (output B,C,H,W)
        ViT  →  model.blocks[-1].norm1    (output B,N,C — NOT blocks[-1];
                that output has zero patch gradients as CLS is the only
                token used for classification)
    reshape_transform : callable | None
        Pass vit_reshape_transform for ViT; None for CNNs.
    """

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.model.eval()
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Returns (heatmap, predicted_class_index).
        heatmap is (H, W) numpy float32 in [0, 1].
        """
        with torch.enable_grad():
            inp = input_tensor.clone().requires_grad_(True)
            output = self.model(inp)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[0, target_class].backward()

        act  = self.activations
        grad = self.gradients

        if self.reshape_transform is not None:
            act  = self.reshape_transform(act)
            grad = self.reshape_transform(grad)

        # Global-average-pool gradients over spatial dims → channel weights
        weights = grad.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        cam = (weights * act).sum(dim=1).squeeze()              # (H, W)
        cam = np.maximum(cam.cpu().numpy(), 0)                  # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


# ─────────────────────────────────────────────
#  ATTENTION ROLLOUT  (timm ViT)
# ─────────────────────────────────────────────

class AttentionRolloutExtractor:
    """
    Attention Rollout for timm ViT (Abnar & Zuidema 2020).

    Hooks attn_drop in every transformer block to capture the post-softmax
    attention matrix (B, heads, N, N).  Rolls up across layers with an
    identity residual and row normalisation as required by the paper.

    fused_attn is disabled on construction so attention weights are
    materialised (PyTorch's scaled_dot_product_attention does not expose them).

    Parameters
    ----------
    model        : timm ViT
    head_fusion  : 'mean' | 'max' | 'min'
    discard_ratio: float in [0, 1) — zero out this fraction of lowest
                   attention values per layer to reduce background noise.
    """

    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.model.eval()
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self._hooks = []

        # Must disable fused attention — SDPA does not expose the attn matrix
        for block in self.model.blocks:
            block.attn.fused_attn = False
            h = block.attn.attn_drop.register_forward_hook(self._save_attn)
            self._hooks.append(h)

    def _save_attn(self, module, input, output):
        # output: (B, heads, N, N)  post-softmax, rows sum to 1
        self.attentions.append(output.detach().cpu())

    @torch.no_grad()
    def generate(self, input_tensor):
        """
        Returns (heatmap, predicted_class_index).
        heatmap is (H, W) numpy float32 in [0, 1].
        """
        self.attentions = []
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        if not self.attentions:
            raise RuntimeError("No attention weights captured — is fused_attn disabled?")

        N = self.attentions[0].shape[-1]                       # 197 for ViT-B/16
        rollout = torch.eye(N)                                 # identity = residual start

        for attn in self.attentions:
            # Fuse heads
            if self.head_fusion == 'mean':
                a = attn.mean(dim=1)[0]                        # (N, N)
            elif self.head_fusion == 'max':
                a = attn.max(dim=1).values[0]
            else:
                a = attn.min(dim=1).values[0]

            # Optional: discard low-attention values to reduce noise
            if self.discard_ratio > 0:
                flat = a.flatten()
                k = int(flat.size(0) * self.discard_ratio)
                threshold = flat.kthvalue(k).values
                a = torch.where(a < threshold, torch.zeros_like(a), a)

            # Add identity residual (required by the paper: A_hat = A + I)
            a = a + torch.eye(N)

            # Row-normalise so rows remain probability distributions
            a = a / a.sum(dim=-1, keepdim=True)

            # Left-multiply to accumulate rollout
            rollout = torch.matmul(a, rollout)

        # CLS row, patch columns (skip col 0 = cls self-attention)
        mask = rollout[0, 1:].numpy()                          # (196,)
        grid = int(mask.shape[0] ** 0.5)                       # 14
        mask = mask.reshape(grid, grid)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask, pred_class


# ─────────────────────────────────────────────
#  OVERLAY UTILITY
# ─────────────────────────────────────────────

def overlay_heatmap(original, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Blend a (H,W) float heatmap in [0,1] onto a PIL image.

    Parameters
    ----------
    original : PIL Image
    heatmap  : (h, w) numpy float32 in [0, 1]
    alpha    : blend weight (0 = only original, 1 = only heatmap)

    Returns
    -------
    PIL Image
    """
    w, h = original.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_np = np.array(original.convert("RGB"))
    blended     = np.uint8(original_np * (1 - alpha) + heatmap_color * alpha)
    return Image.fromarray(blended)
