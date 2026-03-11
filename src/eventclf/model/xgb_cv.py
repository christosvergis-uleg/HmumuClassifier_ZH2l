from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib
import xgboost as xgb

from .base import _prob1

@dataclass
class XGBoostCVClassifier:
    """
    Cross-validation XGBoost classifier using a precomputed fold id per event

    Stores:
      - models_: one XGBClassifier per fold
      - oof_pred_: out-of-fold P(class=1) per training event
      - fold_metrics_: lightweight per-fold info (optional)
      - evals_result_: xgboost training curves per fold (optional)
    """

    xgb_params: Dict[str,Any]
    n_folds: int = 4
    threshold: float = 0.5
    store_evals: bool = True

    models_: List[xgb.XGBClassifier] = field(default_factory=list, init=False)
    oof_pred_: Optional[np.ndarray] = field(default=None, init=False)
    fold_metrics_: List[ Dict[str , Any] ] = field(default_factory=list,init=False)
    evals_result_: List[ Dict[str , Any] ] = field(default_factory=list,init=False)

    def fit(self, 
            X:np.ndarray, y:np.ndarray, w:Optional[np.ndarray]=None, 
            fold_id:Optional[np.ndarray]=None,*, 
            eval_set:bool=True, verbose:bool=False,
            early_stopping_rounds:Optional[int]=None)-> "XGBoostCVClassifier":
        """
        Train n_folds models. Requires fold_id with values in [0, n_folds-1].
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int64,copy=False)
        w_arr = None if w is None else np.asarray(w).astype(np.float64, copy=False)

        if fold_id is None:
            raise ValueError(f"fold id is required, eg event_id%n_folds.")
        fold_id = np.asarray(fold_id).astype(np.int64,copy=False)
        
        if np.any((fold_id<0) | (fold_id>=self.n_folds)):
            bad = np.unique(fold_id[(fold_id < 0) | (fold_id >= self.n_folds)])
            raise ValueError(f"fold_id contains values outside [0,{self.n_folds-1}]: {bad}")
        
        n = X.shape[0]
        self.oof_pred_ = np.full(n, np.nan, dtype=float)
        self.models_.clear()
        self.fold_metrics_.clear()
        self.evals_result_.clear()

        for k in range(self.n_folds):
            tr_idx  = np.where(fold_id != k)
            val_idx = np.where(fold_id == k)

            X_tr , y_tr  = X[tr_idx]  , y[tr_idx]
            X_val, y_val = X[val_idx] , y[val_idx]
            w_tr = w_arr[tr_idx] if w_arr is not None else None 
            w_val= w_arr[val_idx] if w_arr is not None else None 

            model = xgb.XGBClassifier(**self.xgb_params, use_label_encoder=False)
            fit_kwargs: Dict[str, Any] = {"verbose": verbose}

            if w_tr is not None: 
                fit_kwargs["sample_weight"] = w_tr 

            if eval_set:
                eval_list = [(X_tr,y_tr), (X_val,y_val)]
                fit_kwargs["eval_set"] = eval_list
                if early_stopping_rounds is not None:
                    fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

            model.fit(X_tr, y_tr, **fit_kwargs)

            #Validation prediction = out of fold prediction for those indices
            pred_val = _prob1(model.predict_proba(X_val))
            self.oof_pred_[val_idx] = pred_val

            # Store curves if requested
            if self.store_evals and eval_set:
                # xgb stores eval history inside the model
                self.evals_result_.append(model.evals_result())
            else:
                self.evals_result_.append({})

            # Minimal per-fold summary (you can swap this for your metrics module later)
            fold_info = {
                "fold": int(k),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(val_idx)),
                "valid_score_mean": float(np.average(pred_val, weights=w_val) if w_val is not None else np.mean(pred_val)),
            }
            self.fold_metrics_.append(fold_info)

            self.models_.append(model)

        if np.isnan(self.oof_pred_).any():
            missing = int(np.isnan(self.oof_pred_).sum())
            raise RuntimeError(f"OOF predictions not filled for {missing} events. Check fold_id logic.")

        return self
    
    def predict_proba(self, X:np.ndarray, agg:str="mean")-> np.ndarray:
        """
        Predict P(class=1) by aggregating per-fold models.
        
        agg: 'mean' or 'median'
        """

        if not self.models_:
           raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        preds = np.vstack([_prob1(m.predict_proba(X)) for m in self.models_])  # (n_folds, n)

        if agg == "mean":
            return preds.mean(axis=0)
        if agg == "median":
            return np.median(preds, axis=0)
        raise ValueError(f"Unknown agg='{agg}'")

    def predict_proba_by_fold(self, X: np.ndarray, fold: int) -> np.ndarray:
        """Use a specific fold model (useful for your 'apply model to same-fold subset' pattern)."""
        if not self.models_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if fold < 0 or fold >= len(self.models_):
            raise ValueError(f"fold must be in [0,{len(self.models_)-1}]")
        return _prob1(self.models_[fold].predict_proba(np.asarray(X)))
    
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "XGBoostCVClassifier":
        obj = joblib.load(path)
        if not isinstance(obj, XGBoostCVClassifier):
            raise TypeError(f"Loaded object is not XGBoostCVClassifier: {type(obj)}")
        return obj