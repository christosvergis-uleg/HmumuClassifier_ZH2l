from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

from eventclf.model.xgb_cv import XGBoostCVClassifier
from eventclf.eval.metrics import evaluate_binary_classifier


def xgb_cv_objective(params: Dict[str, Any],*,
    X: np.ndarray,y: np.ndarray,w: Optional[np.ndarray],fold_id: np.ndarray,
    metric: str = "auc",   # "auc" or "logloss"
    n_folds: int = 4,) -> float:
    """
    Returns a scalar score for hyperparameter tuning.
    Higher is better (so for logloss we return -logloss).
    """

    cv = XGBoostCVClassifier(xgb_params=params,n_folds=n_folds,store_evals=False,)

    cv.fit(X=X, y=y, w=w, fold_id=fold_id, eval_set=False, verbose=False, )

    metrics = evaluate_binary_classifier(y_true=y, y_score=cv.oof_pred_, sample_weight=w)

    if metric == "auc":
        return float(metrics["auc"])
    if metric == "logloss":
        return -float(metrics["logloss"])  # maximize negative loss
    raise ValueError(f"Unknown metric: {metric}")
