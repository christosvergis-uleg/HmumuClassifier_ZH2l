from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp

def psi(ref: np.ndarray, new: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    ref = np.asarray(ref, dtype=float)
    new = np.asarray(new, dtype=float)
    ref = ref[np.isfinite(ref)]
    new = new[np.isfinite(new)]
    if ref.size == 0 or new.size == 0:
        return float("nan")

    q = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(ref, q)
    bins[0], bins[-1] = -np.inf, np.inf

    r = np.histogram(ref, bins=bins)[0].astype(float)
    n = np.histogram(new, bins=bins)[0].astype(float)

    r = np.clip(r / max(r.sum(), 1.0), eps, 1.0)
    n = np.clip(n / max(n.sum(), 1.0), eps, 1.0)
    return float(np.sum((n - r) * np.log(n / r)))

def ks_pvalue(ref: np.ndarray, new: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=float)
    new = np.asarray(new, dtype=float)
    ref = ref[np.isfinite(ref)]
    new = new[np.isfinite(new)]
    if ref.size == 0 or new.size == 0:
        return float("nan")
    return float(ks_2samp(ref, new).pvalue)
