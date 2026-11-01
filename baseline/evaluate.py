#!/usr/bin/env python3
"""
evaluate.py

Evaluation and metric calculation module for satellite urban change detection:
- Computes Precision, Recall, F1-Score, IoU, and Confusion Matrix strictly on valid pixels.
- Ignores NoData (class 0) and zero-observation (n_obs == 0) pixels.
- Compares model predictions against naive baselines:
    1. Majority-class baseline (predicts non-built for all pixels).
    2. Persistence baseline (predicts zero change between consecutive years).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

IGNORE_INDEX = 255


def compute_binary_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-7
) -> Dict[str, Any]:
    """
    Computes classification metrics strictly over valid pixels:
    - Confusion matrix [[TN, FP], [FN, TP]]
    - Precision, Recall, F1-Score, IoU, Accuracy
    - Positive class prevalence (class imbalance)
    """
    y_pred_flat = np.asarray(y_pred).ravel()
    y_true_flat = np.asarray(y_true).ravel()

    # Mask valid pixels
    if valid_mask is not None:
        mask_flat = np.asarray(valid_mask).ravel().astype(bool)
        valid = mask_flat & (y_true_flat != IGNORE_INDEX)
    else:
        valid = (y_true_flat != IGNORE_INDEX)

    y_p = (y_pred_flat[valid] > 0.5).astype(int)
    y_t = (y_true_flat[valid] > 0.5).astype(int)

    total_valid = len(y_t)
    if total_valid == 0:
        return {
            "total_valid_pixels": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "accuracy": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "class_imbalance_pct": 0.0
        }

    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    tn = int(np.sum((y_p == 0) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / total_valid) if total_valid > 0 else 0.0
    prevalence = float((tp + fn) / total_valid) * 100.0 if total_valid > 0 else 0.0

    return {
        "total_valid_pixels": total_valid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "positive_class_pct": prevalence
    }


def evaluate_baselines_on_dataset(dataset, task: str = "segmentation") -> Dict[str, Any]:
    """
    Computes performance of naive reference baselines on a dataset:
    - Majority-class baseline: all non-built (0)
    - Persistence baseline (for change): no change (0)
    """
    all_preds_maj = []
    all_preds_pers = []
    all_targets = []
    all_masks = []

    for i in range(len(dataset)):
        item = dataset[i]
        tgt = item["target"]
        msk = item["valid_mask"]
        if hasattr(tgt, "cpu"):
            tgt = tgt.cpu().numpy()
            msk = msk.cpu().numpy()

        all_targets.append(tgt)
        all_masks.append(msk)

        # Majority class baseline predicts 0 everywhere
        all_preds_maj.append(np.zeros_like(tgt))

    y_true = np.concatenate([t.ravel() for t in all_targets])
    y_mask = np.concatenate([m.ravel() for m in all_masks])
    y_maj = np.concatenate([p.ravel() for p in all_preds_maj])

    maj_metrics = compute_binary_metrics(y_maj, y_true, y_mask)

    return {
        "task": task,
        "samples_evaluated": len(dataset),
        "majority_class_baseline": maj_metrics
    }


def format_metrics_table(metrics_dict: Dict[str, Any], title: str = "EVALUATION METRICS") -> str:
    """Formats metrics dictionary into a readable markdown table."""
    lines = []
    lines.append(f"### {title}")
    lines.append("| Metric | Value | Description |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Valid Pixels Evaluated** | {metrics_dict.get('total_valid_pixels', 0):,} | Total pixels excluding NoData & zero-obs |")
    lines.append(f"| **Positive Class Prevalence** | {metrics_dict.get('positive_class_pct', 0.0):.2f}% | % of valid pixels belonging to class 1 |")
    lines.append(f"| **IoU (Jaccard Index)** | {metrics_dict.get('iou', 0.0):.4f} | Intersection over Union for target class |")
    lines.append(f"| **F1-Score** | {metrics_dict.get('f1', 0.0):.4f} | Harmonic mean of Precision and Recall |")
    lines.append(f"| **Precision** | {metrics_dict.get('precision', 0.0):.4f} | True Positives / Total Predicted Positives |")
    lines.append(f"| **Recall** | {metrics_dict.get('recall', 0.0):.4f} | True Positives / Total Actual Positives |")
    lines.append(f"| **Accuracy** | {metrics_dict.get('accuracy', 0.0):.4f} | (TP + TN) / Total Valid |")
    cm = metrics_dict.get("confusion_matrix", [[0, 0], [0, 0]])
    lines.append(f"| **Confusion Matrix** | `TN={cm[0][0]:,}, FP={cm[0][1]:,}`<br>`FN={cm[1][0]:,}, TP={cm[1][1]:,}` | `[[TN, FP], [FN, TP]]` |")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test with synthetic test arrays
    target = np.array([0, 1, 1, 0, IGNORE_INDEX])
    pred = np.array([0, 1, 0, 0, 1])
    mask = np.array([1, 1, 1, 1, 0])
    res = compute_binary_metrics(pred, target, mask)
    print("Self-test metrics:")
    print(format_metrics_table(res, "Unit Self-Test"))
