from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging
import time

import numpy as np
import xgboost as xgb
from packaging import version
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from .base import _prob1

log = logging.getLogger(__name__)


@dataclass
class XGBRotatingTrainValBlindTrainer:
    """
    Rotating blind trainer with a FIXED validation fold per outer split.

    For each outer fold k:
      - test fold = k
      - validation fold = val_fold_fn(k) or default (k - 1) % n_folds
      - training folds = all remaining folds

    Hyperparameter tuning is done using ONE fixed validation fold only,
    matching the diagram:
      3 folds train + 1 fold validation + 1 fold test

    After tuning, the final model is fit on:
      - train only, or
      - train + validation
    depending on refit_on_train_val.

    The blind test fold is never used in tuning or fitting.
    """

    base_params: Dict[str, Any]
    param_distributions: Dict[str, Any]
    n_folds: int = 5
    n_iter: int = 20
    scoring: str = "average_precision"   # "average_precision" or "roc_auc"
    random_state: int = 42
    verbose: int = 0
    refit_on_train_val: bool = True

    models_: List[xgb.XGBClassifier] = field(default_factory=list, init=False)
    best_params_per_fold_: List[Dict[str, Any]] = field(default_factory=list, init=False)
    blind_pred_: Optional[np.ndarray] = field(default=None, init=False)
    fold_summary_: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def _score_metric(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> float:
        if self.scoring == "average_precision":
            return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))
        if self.scoring == "roc_auc":
            return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
        raise ValueError(
            f"Unsupported scoring='{self.scoring}'. "
            "Use 'average_precision' or 'roc_auc'."
        )

    def _default_val_fold(self, test_fold: int) -> int:
        return (test_fold - 1) % self.n_folds

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray],
        fold_id: np.ndarray,
    ) -> "XGBRotatingTrainValBlindTrainer":
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int64, copy=False)
        w_arr = None if w is None else np.asarray(w).astype(np.float64, copy=False)
        fold_id = np.asarray(fold_id).astype(np.int64, copy=False)

        if np.any((fold_id < 0) | (fold_id >= self.n_folds)):
            raise ValueError(f"fold_id must be in [0, {self.n_folds - 1}]")

        n = X.shape[0]
        self.blind_pred_ = np.full(n, np.nan, dtype=float)
        self.models_.clear()
        self.best_params_per_fold_.clear()
        self.fold_summary_.clear()

        for k in range(self.n_folds):
            t0 = time.time()

            sampled_params = list(
                ParameterSampler(
                    self.param_distributions,
                    n_iter=self.n_iter,
                    random_state=self.random_state + k,
                )
            )
            log.info("[Fold %d/%d] Sampling %d hyperparameter configurations:", k + 1, self.n_folds, len(sampled_params))
            for i, p in enumerate(sampled_params):
                log.info("  [%03d] %s", i, p)

            test_fold = k
            val_fold = self._default_val_fold(test_fold)

            train_mask = (fold_id != test_fold) & (fold_id != val_fold)
            val_mask = (fold_id == val_fold)
            test_mask = (fold_id == test_fold)

            tr_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            te_idx = np.where(test_mask)[0]

            if len(tr_idx) == 0 or len(val_idx) == 0 or len(te_idx) == 0:
                raise RuntimeError(
                    f"Empty split for test_fold={test_fold}, val_fold={val_fold}: "
                    f"n_train={len(tr_idx)}, n_val={len(val_idx)}, n_test={len(te_idx)}"
                )

            log.info("[Fold %d/%d] Starting rotating train/val/blind step", k + 1, self.n_folds)
            log.info(
                "[Fold %d/%d] Train folds: all except val=%d and test=%d",
                k + 1, self.n_folds, val_fold, test_fold
            )
            log.info(
                "[Fold %d/%d] Train size: %d | Validation size: %d | Test size: %d",
                k + 1, self.n_folds, len(tr_idx), len(val_idx), len(te_idx)
            )

            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xval, yval = X[val_idx], y[val_idx]
            Xte, yte = X[te_idx], y[te_idx]

            wtr = w_arr[tr_idx] if w_arr is not None else None
            wval = w_arr[val_idx] if w_arr is not None else None
            wte = w_arr[te_idx] if w_arr is not None else None

            # --------------------------------------------------
            # Tune on one fixed validation fold only
            # --------------------------------------------------
            best_score = -np.inf
            best_params = None

            log.info(
                "[Fold %d/%d] Starting hyperparameter tuning (%d sampled parameter sets, fixed validation fold)",
                k + 1, self.n_folds, len(sampled_params)
            )

            for i, sampled in enumerate(sampled_params, start=1):
                params = dict(self.base_params)
                params.update(sampled)

                if version.parse(xgb.__version__) < version.parse("1.3.0"):
                    model = xgb.XGBClassifier(**params, use_label_encoder=False)
                else:
                    model = xgb.XGBClassifier(**params)

                fit_kwargs: Dict[str, Any] = {"verbose": False}
                if wtr is not None:
                    fit_kwargs["sample_weight"] = wtr

                model.fit(Xtr, ytr, **fit_kwargs)
                p_val = _prob1(model.predict_proba(Xval))
                score = self._score_metric(yval, p_val, wval)

                if self.verbose:
                    log.info(
                        "[Fold %d/%d] Candidate %d/%d | %s = %.6f",
                        k + 1, self.n_folds, i, len(sampled_params), self.scoring, score
                    )

                if score > best_score:
                    best_score = score
                    best_params = params

            if best_params is None:
                raise RuntimeError("Hyperparameter tuning failed to select best params.")

            self.best_params_per_fold_.append(best_params)

            log.info(
                "[Fold %d/%d] Best validation %s: %.6f",
                k + 1, self.n_folds, self.scoring, best_score
            )
            log.info(
                "[Fold %d/%d] Best params: %s",
                k + 1, self.n_folds, best_params
            )

            # --------------------------------------------------
            # Final refit
            # --------------------------------------------------
            if self.refit_on_train_val:
                Xfit = np.concatenate([Xtr, Xval], axis=0)
                yfit = np.concatenate([ytr, yval], axis=0)
                if wtr is not None and wval is not None:
                    wfit = np.concatenate([wtr, wval], axis=0)
                else:
                    wfit = None
                log.info("[Fold %d/%d] Refitting final model on train + validation", k + 1, self.n_folds)
            else:
                Xfit, yfit, wfit = Xtr, ytr, wtr
                log.info("[Fold %d/%d] Refitting final model on training only", k + 1, self.n_folds)

            if version.parse(xgb.__version__) < version.parse("1.3.0"):
                final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
            else:
                final_model = xgb.XGBClassifier(**best_params)

            fit_kwargs = {"verbose": False}
            if wfit is not None:
                fit_kwargs["sample_weight"] = wfit

            final_model.fit(Xfit, yfit, **fit_kwargs)

            # --------------------------------------------------
            # Blind test prediction
            # --------------------------------------------------
            log.info("[Fold %d/%d] Predicting on blind test fold", k + 1, self.n_folds)
            p_te = _prob1(final_model.predict_proba(Xte))
            self.blind_pred_[te_idx] = p_te
            self.models_.append(final_model)

            fold_auc = roc_auc_score(yte, p_te, sample_weight=wte) if wte is not None else roc_auc_score(yte, p_te)
            fold_ap = average_precision_score(yte, p_te, sample_weight=wte) if wte is not None else average_precision_score(yte, p_te)

            self.fold_summary_.append(
                {
                    "fold": int(k),
                    "test_fold": int(test_fold),
                    "val_fold": int(val_fold),
                    "n_train": int(len(tr_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(te_idx)),
                    "best_val_score": float(best_score),
                    "blind_auc": float(fold_auc),
                    "blind_ap": float(fold_ap),
                }
            )

            log.info(
                "[Fold %d/%d] Blind-fold metrics | AUC=%.6f | AP=%.6f",
                k + 1, self.n_folds, fold_auc, fold_ap
            )
            log.info(
                "[Fold %d/%d] Completed in %.1f seconds",
                k + 1, self.n_folds, time.time() - t0
            )

        if np.isnan(self.blind_pred_).any():
            raise RuntimeError("blind_pred_ contains NaNs. Check fold coverage.")

        return self