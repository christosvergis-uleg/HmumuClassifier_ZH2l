from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from packaging import version

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

from eventclf.monitoring import plot_train_test_feature_distributions

from .base import _prob1
import logging
log = logging.getLogger(__name__)
import time

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
    calib_mod_method: str = "sigmoid"

    models_: List[xgb.XGBClassifier] = field(default_factory=list, init=False)
    best_params_per_fold_: List[Dict[str, Any]] = field(default_factory=list, init=False)
    blind_pred_: Optional[np.ndarray] = field(default=None, init=False)
    feature_plot_paths_: List[str] = field(default_factory=list, init=False)


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray],
        fold_id: np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
        feature_plot_dir: Optional[str | Path] = None,
        feature_plot_bins: int = 50,
        feature_plot_density: bool = False,
        feature_plot_normalize: bool = True,
        feature_plot_ncols: int = 2,
        feature_plot_config: Optional[Dict[str, Any]] = None,
        atlas_label: str = "ATLAS-Internal",
        com_energy: str = r"$\sqrt{s}=13.6\ \mathrm{TeV}$",
        lumi: str = r"$165\ \mathrm{fb}^{-1}$",
        show_atlas_label: bool = True,
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
            t0 = time.time()
            log.info("[Fold %d/%d] Starting rotating blind step", k  + 1, self.n_folds)
            tr_idx = np.where(fold_id != k)[0]
            te_idx = np.where(fold_id == k)[0]
            log.info("[Fold %d/%d] Using fold %d as blind test set", k + 1, self.n_folds, k)

            log.info("[Fold %d/%d] Train size: %d | Test size: %d", k + 1, self.n_folds, 
                len(tr_idx),
                len(te_idx),
            )

            Xtr, ytr = X[tr_idx], y[tr_idx]
            wtr = w_arr[tr_idx] if w_arr is not None else None

            Xte, yte = X[te_idx], y[te_idx]
            wte = w_arr[te_idx] if w_arr is not None else None

            # --- Tune only on training portion (4 folds) ---
            if version.parse(xgb.__version__) < version.parse("1.3.0"):
                estimator = xgb.XGBClassifier(**self.base_params, use_label_encoder=False)
            else: 
                estimator = xgb.XGBClassifier(**self.base_params)
 
            log.info("[Fold %d/%d] Starting hyperparameter tuning (%d iterations, %d-fold CV)",
                k + 1, self.n_folds, self.n_iter, self.inner_cv,)
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
            log.info( "[Fold %d/%d] Best params: %s", k + 1, self.n_folds, rs.best_params_,)

            log.info( "[Fold %d/%d] Best score: %.6f", k + 1, self.n_folds, rs.best_score_,)

            best_params = dict(self.base_params)
            best_params.update(dict(rs.best_params_))
            self.best_params_per_fold_.append(best_params)

            # --- Train final model on all 4 folds (no peeking at fold k) ---
            model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
            log.info("[Fold %d/%d] Training final model on training folds", k + 1, self.n_folds)
            if wtr is not None:
                model.fit(Xtr, ytr, sample_weight=wtr, verbose=False)
            else:
                model.fit(Xtr, ytr, verbose=False)

            #log.info("[Fold %d/%d] Fitting calibrated model with method='%s'", k + 1, self.n_folds, self.calib_mod_method)
            #cal_model = CalibratedClassifierCV(estimator=model, method=self.calib_mod_method, cv=3,)
            #if wtr is not None:
            #    cal_model.fit(Xtr, ytr, sample_weight=wtr)
            #else:
            #    cal_model.fit(Xtr, ytr)

            # --- Blind prediction on held-out fold k ---
            log.info("[Fold %d/%d] Predicting on blind fold", k + 1, self.n_folds)
            p_te = _prob1(model.predict_proba(Xte))
            self.blind_pred_[te_idx] = p_te

            self.models_.append(model)

            if feature_plot_dir is not None and feature_names is not None:
                plot_path = Path(feature_plot_dir) / f"feature_distributions_fold{k}.png"
                plot_train_test_feature_distributions(
                    X_train=Xtr,
                    y_train=ytr,
                    X_test=Xte,
                    y_test=yte,
                    w_train=wtr,
                    w_test=wte,
                    feature_names=feature_names,
                    output_path=plot_path,
                    bins=feature_plot_bins,
                    density=feature_plot_density,
                    normalize=feature_plot_normalize,
                    ncols=feature_plot_ncols,
                    title=f"Input feature comparison, fold {k}",
                    feature_plot_config=feature_plot_config,
                    atlas_label=atlas_label,
                    com_energy=com_energy,
                    lumi=lumi,
                    show_atlas_label=show_atlas_label,
                )
                self.feature_plot_paths_.append(str(plot_path))

 
            if w_arr is not None:
                fold_auc = roc_auc_score(yte, p_te, sample_weight=wte)
                fold_ap = average_precision_score(yte, p_te, sample_weight=wte)
            else:
                fold_auc = roc_auc_score(yte, p_te)
                fold_ap = average_precision_score(yte, p_te)

            log.info(
                "[Fold %d/%d] Blind-fold metrics | AUC=%.6f | AP=%.6f",
                k + 1,
                self.n_folds,
                fold_auc,
                fold_ap,
            )
  
            log.info("[Fold %d/%d] Done", k + 1, self.n_folds)
            log.info("[Fold %d/%d] Completed in %.1f seconds",k + 1, self.n_folds, time.time() - t0,)

        if np.isnan(self.blind_pred_).any():
            raise RuntimeError("blind_pred_ contains NaNs. Check fold_id coverage.")

        return self