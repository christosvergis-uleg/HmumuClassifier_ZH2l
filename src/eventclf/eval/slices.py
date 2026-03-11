from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from .metrics import evaluate_binary_classifier

def evaluate_slices(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    slice_specs: Dict[str, Any],
    weight_col: Optional[str] = None,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, mask in slice_specs.items():
        sub = df.loc[mask]
        w = sub[weight_col].to_numpy() if weight_col else None
        out[name] = evaluate_binary_classifier(sub[y_col], sub[score_col], sample_weight=w, extra={"slice_n": int(len(sub))})
    return out
