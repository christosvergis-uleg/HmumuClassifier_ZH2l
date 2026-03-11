from eventclf.model.xgb_cv import XGBoostCVClassifier
from eventclf.utils.io import load_json

# Your X has FEATURES + ["fold"] in your script :contentReference[oaicite:4]{index=4}
fold_id = X["fold"].to_numpy()
X_feat  = X.drop(columns=["fold"]).to_numpy()
y_arr   = y.to_numpy()
w_arr   = w.to_numpy()

params_best = load_json("configs/xgb_hmumu_zh2l_best_v1.json")["best_params"]
  # you already have this dict :contentReference[oaicite:5]{index=5}

cv = XGBoostCVClassifier(
    xgb_params=params,
    n_folds=4,
    store_evals=True,
)

cv.fit(
    X_feat,
    y_arr,
    w=w_arr,
    fold_id=fold_id,
    eval_set=True,
    verbose=False,
    early_stopping_rounds=None,   # set if you want
)

# This replaces: X_update.loc[val_idx, 'BDT_score'] per fold
dataset_df["BDT_score"] = cv.oof_pred_

# This replaces: dataset_df["BDT_avg"] = mean(bdt_predictions, axis=1)
dataset_df["BDT_avg"] = cv.predict_proba(X_feat, agg="mean")

cv.save("xgb_cv_model.joblib")