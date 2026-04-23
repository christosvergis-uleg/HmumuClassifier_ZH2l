from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    """
    Class defining how to create ML arrays from Dataframes
    """
    features: Sequence[str]
    label: str
    weight: Optional[str]=None
    event_id: Optional[str]= "event_number"
    extra_cols: Sequence[str]=() # e.g. ("met", "Njets") for slice eval

@dataclass
class DatasetArrays:
    X: np.array
    y: np.array 
    w: Optional[np.array]
    meta: Dict[str , np.array] #event_id, extras etc

def _require_columns(df: pd.DataFrame, cols: Sequence[str])->None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}")

def build_arrays(df: pd.DataFrame, spec: DataSpec, dtype: Any =np.float32)-> DatasetArrays:
    """
    Convert a DataFrame into (X, y, w, meta) with consistent column ordering.
    """
    needed = list(spec.features) + [spec.label]
    if spec.weight:
        needed.append( spec.weight)
    if spec.event_id:
        needed.append( spec.event_id)
    needed += list(spec.extra_cols)

    _require_columns(df, needed)

    #Explanatory features
    X = df.loc[:, list(spec.features)].to_numpy(dtype=dtype, copy=True)
    y = df[spec.label].to_numpy(dtype=dtype, copy=True)
    y = y.astype(np.int64, copy=False)

    #Optional weights 
    w = None
    if spec.weight:
        w = df[spec.weight].to_numpy(copy=True).astype(np.float64, copy=False)
    
    # Meta
    meta: Dict[str, np.ndarray] = {}
    if spec.event_id:
        meta["event_id"] = df[spec.event_id].to_numpy(copy=True)
    for col in spec.extra_cols:
        meta[col] = df[col].to_numpy(copy=True)
    return DatasetArrays(X=X, y=y, w=w, meta=meta)

def subset(arr: DatasetArrays, idx: np.ndarray) -> DatasetArrays:
    """
    Convenience function to subset DatasetArrays with indices or boolean mask.
    """
    idx = np.asarray(idx)
    X = arr.X[idx]
    y = arr.y[idx]
    w = arr.w[idx] if arr.w is not None else None
    meta = {k: v[idx] for k, v in arr.meta.items()}
    return DatasetArrays(X=X, y=y, w=w, meta=meta)