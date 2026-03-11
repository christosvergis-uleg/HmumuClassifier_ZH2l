from __future__ import annotations
import numpy as np

def _prob1(p: np.ndarray) -> np.ndarray:
    #Returns Class=1 probability in shape (n,)
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1].astype(float, copy=False)
    if p.ndim == 1:
        return p.astype(float, copy=False)
    raise ValueError(f"Unexpected predict_proba shape: {p.shape}")
