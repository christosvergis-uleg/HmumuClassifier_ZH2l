from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np

def feature_summary(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x)
    if x.size == 0:
        return {"n": 0, "finite_n": 0, "nan_frac": 0.0}

    if np.issubdtype(x.dtype, np.floating):
        finite = np.isfinite(x)
        xf = x[finite]
        nan_frac = float(1.0 - finite.mean())
    else:
        xf = x
        nan_frac = 0.0

    if xf.size == 0:
        return {"n": int(x.size), "finite_n": 0, "nan_frac": nan_frac}

    return {
        "n": int(x.size),
        "finite_n": int(xf.size),
        "nan_frac": nan_frac,
        "min": float(np.min(xf)),
        "max": float(np.max(xf)),
        "mean": float(np.mean(xf)),
        "std": float(np.std(xf)),
        "p01": float(np.quantile(xf, 0.01)),
        "p50": float(np.quantile(xf, 0.50)),
        "p99": float(np.quantile(xf, 0.99)),
    }

def assert_feature_contract(summary: Dict[str, Any], 
                            allow_nan_frac: float = 0.0,
                            bounds: Optional[Tuple[float, float]] = None,) -> None:
    if summary.get("nan_frac", 0.0) > allow_nan_frac:
        raise AssertionError(f"nan_frac {summary['nan_frac']:.4f} > {allow_nan_frac}")

    if bounds is not None and "p01" in summary and "p99" in summary:
        lo, hi = bounds
        if summary["p01"] < lo or summary["p99"] > hi:
            raise AssertionError(f"quantiles out of bounds: p01={summary['p01']:.4g}, p99={summary['p99']:.4g}, bounds={bounds}")
