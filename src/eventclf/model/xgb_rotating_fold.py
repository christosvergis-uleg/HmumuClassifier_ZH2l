from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from packaging import version

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from .base import _prob1

@dataclass
class XGBRotatingBlindTrainer:
    """
    Your exact scheme:

    - Split data into K folds (e.g. K=5).
    - For each held-out fold k:
        - tune hyperparameters using RandomizedSearchCV(cv=K-1) on the other folds only
        - fit final model on the other folds
        - predict on the held-out fold (blind)
    - Produces one blind score per event (OOF-style), plus per-fold models/params.

    Key property:
      The held-out fold is NEVER used during tuning or fitting.
      (Unless you explicitly add it yourself later.)
    """

    base_params: Dict[str, Any]
    param_distributions: Dict[str, Any]
    n_folds: int = 5                  # outer folds
    inner_cv: int = 4                 # RandomSearchCV folds on the training part
    n_iter: int = 100
    scoring: str = "neg_log_loss"
    random_state: int = 42
    verbose: int = 0

    models_: List[xgb.XGBClassifier] = field(default_factory=list, init=False)
    best_params_per_fold_: List[Dict[str, Any]] = field(default_factory=list, init=False)
    blind_pred_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray],
        fold_id: np.ndarray,
    ) -> "XGBRotatingBlindTrainer":
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int64, copy=False)
        w_arr = None if w is None else np.asarray(w).astype(np.float64, copy=False)
        fold_id = np.asarray(fold_id).astype(np.int64, copy=False)

        if np.any((fold_id < 0) | (fold_id >= self.n_folds)):
            raise ValueError(f"fold_id must be in [0, {self.n_folds-1}]")

        n = X.shape[0]
        self.blind_pred_ = np.full(n, np.nan, dtype=float)
        self.models_.clear()
        self.best_params_per_fold_.clear()

        for k in range(self.n_folds):
            tr_idx = np.where(fold_id != k)[0]
            te_idx = np.where(fold_id == k)[0]

            Xtr, ytr = X[tr_idx], y[tr_idx]
            wtr = w_arr[tr_idx] if w_arr is not None else None

            # --- Tune only on training portion (4 folds) ---
            if version.parse(xgb.__version__) < version.parse("1.3.0"):
                estimator = xgb.XGBClassifier(**self.base_params, use_label_encoder=False)
            else: 
                estimator = xgb.XGBClassifier(**self.base_params)

            rs = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=self.param_distributions,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.inner_cv,
                random_state=self.random_state + k,
                verbose=self.verbose,
            )
            if wtr is not None:
                rs.fit(Xtr, ytr, sample_weight=wtr)
            else:
                rs.fit(Xtr, ytr)

            best_params = dict(self.base_params)
            best_params.update(dict(rs.best_params_))
            self.best_params_per_fold_.append(best_params)

            # --- Train final model on all 4 folds (no peeking at fold k) ---
            model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
            if wtr is not None:
                model.fit(Xtr, ytr, sample_weight=wtr, verbose=False)
            else:
                model.fit(Xtr, ytr, verbose=False)

            # --- Blind prediction on held-out fold k ---
            p_te = _prob1(model.predict_proba(X[te_idx]))
            self.blind_pred_[te_idx] = p_te

            self.models_.append(model)

        if np.isnan(self.blind_pred_).any():
            raise RuntimeError("blind_pred_ contains NaNs. Check fold_id coverage.")

        return self