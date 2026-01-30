"""
Evaluation metrics and performance analysis utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics

def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    Calculate metrics for each class.
    
    Returns:
        Dictionary mapping class names to their metrics
    """
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    
    per_class = {}
    for class_name in class_names:
        if class_name in report:
            per_class[class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }
    
    return per_class

def calculate_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize by true class
    
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def calculate_map(detections, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        detections: List of detections [(confidence, bbox), ...]
        ground_truths: List of ground truth bboxes
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        mAP score
    """
    # Sort detections by confidence
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    
    true_positives = []
    false_positives = []
    matched_gt = set()
    
    for conf, det_bbox in detections:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_bbox in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
            matched_gt.add(best_gt_idx)
        else:
            true_positives.append(0)
            false_positives.append(1)
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (len(ground_truths) + 1e-10)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap

def print_metrics_summary(metrics, title="Metrics Summary"):
    """Pretty print metrics."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:20s}: {value:.4f}")
        else:
            print(f"{metric_name:20s}: {value}")
    print("=" * 50 + "\n")
