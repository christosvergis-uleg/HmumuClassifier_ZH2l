from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, accuracy_score

def _safe_auc(y_true, y_score, sample_weight=None) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))

def expected_calibration_error(y_true, y_prob, n_bins: int = 15, sample_weight=None) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    w = np.ones_like(y_prob) if sample_weight is None else np.asarray(sample_weight).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total_w = w.sum()
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        ww = w[mask]
        conf = np.average(y_prob[mask], weights=ww)
        acc = np.average(y_true[mask], weights=ww)
        ece += (ww.sum() / total_w) * abs(acc - conf)
    return float(ece)

def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    sample_weight: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    y_pred = (y_score >= threshold).astype(int)
    y_score_clip = np.clip(y_score, 1e-8, 1 - 1e-8)

    metrics: Dict[str, Any] = {
        "n": int(len(y_true)),
        "pos_rate": float(np.average(y_true, weights=sample_weight) if sample_weight is not None else y_true.mean()),
        "auc": _safe_auc(y_true, y_score, sample_weight),
        "ap": float(average_precision_score(y_true, y_score, sample_weight=sample_weight)) if len(np.unique(y_true)) > 1 else float("nan"),
        "logloss": float(log_loss(y_true, y_score_clip, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y_true, y_score, sample_weight=sample_weight)),
        "ece": expected_calibration_error(y_true, y_score, sample_weight=sample_weight),
        "acc": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "threshold": float(threshold),
    }
    if extra:
        metrics.update(extra)
    return metrics
