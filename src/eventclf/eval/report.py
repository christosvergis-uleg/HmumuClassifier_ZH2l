from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from eventclf.utils.io import save_json

def summarize_folds(fold_metrics: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if not fold_metrics:
        raise ValueError("fold_metrics is empty")

    if keys is None:
        # infer numeric keys
        keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]

    out: Dict[str, Any] = {"n_folds": len(fold_metrics)}
    for k in keys:
        vals = [m.get(k) for m in fold_metrics]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if not vals:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
            out[f"{k}_min"] = float("nan")
            continue
        a = np.asarray(vals, dtype=float)
        out[f"{k}_mean"] = float(a.mean())
        out[f"{k}_std"] = float(a.std())
        out[f"{k}_min"] = float(a.min())
    return out

def save_fold_report(path: str, fold_metrics: List[Dict[str, Any]]) -> None:
    save_json(path, {"fold_metrics": fold_metrics, "summary": summarize_folds(fold_metrics)})
