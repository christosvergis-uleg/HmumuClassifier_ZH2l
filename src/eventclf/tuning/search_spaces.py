from __future__ import annotations

def xgb_search_space_v1():
    """
    A reasonable starting search space for physics BDT-like problems.
    Returns dict-like structure suitable for your search method.
    """
    return {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1.0, 2.0, 5.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.5],
        "n_estimators": [400, 800, 1500],
        "tree_method": ["hist"],
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
    }