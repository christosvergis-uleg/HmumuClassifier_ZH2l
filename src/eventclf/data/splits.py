from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold


IndexSplit = List[Tuple[np.ndarray, np.ndarray]]
@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 4
    shuffle: bool = True
    random_state: int = 42


def folds_from_event_mod(event_id: np.ndarray, n_splits: int) -> IndexSplit:
    """
    Deterministic folds based on event_id % n_splits.
    Returns list of (train_idx, valid_idx).
    """
    event_id = np.asarray(event_id)
    if event_id.ndim != 1:
        raise ValueError("event_id must be a 1D array")

    all_idx = np.arange(event_id.shape[0])
    fold_id = np.mod(event_id.astype(np.int64, copy=False), n_splits)

    splits: IndexSplit = []
    for k in range(n_splits):
        valid_idx = all_idx[fold_id == k]
        train_idx = all_idx[fold_id != k]
        splits.append((train_idx, valid_idx))
    return splits

def stratified_folds(y: np.ndarray, cfg: CVConfig) -> IndexSplit:
    """
    Standard StratifiedKFold splits.
    """
    y = np.asarray(y).astype(np.int64, copy=False)
    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.random_state,
    )

    # X not needed; pass dummy
    dummy = np.zeros_like(y)
    return [(tr, va) for tr, va in skf.split(dummy, y)]

def choose_folds(
    *,
    y: np.ndarray,
    event_id: Optional[np.ndarray] = None,
    method: str = "event_mod",
    cfg: CVConfig = CVConfig(),
) -> IndexSplit:
    """
    Convenience wrapper. method in {"event_mod", "stratified"}.
    """
    if method == "event_mod":
        if event_id is None:
            raise ValueError("event_id is required for method='event_mod'")
        return folds_from_event_mod(event_id=event_id, n_splits=cfg.n_splits)

    if method == "stratified":
        return stratified_folds(y=y, cfg=cfg)

    raise ValueError(f"Unknown method: {method}")