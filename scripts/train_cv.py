from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from eventclf.utils.io import load_json, save_json, ensure_dir
from eventclf.config import HMUMU_ZH2L_SCHEMA
from eventclf.io import RootReader
from eventclf.data import DatasetSpec, build_arrays
from eventclf.model.xgb_cv import XGBoostCVClassifier
from eventclf.eval import evaluate_binary_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="Input ROOT files")
    ap.add_argument("--tree", default="tree_Hmumu", help="TTree name")
    ap.add_argument("--config", required=True, help="Path to JSON with XGB params")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    ap.add_argument("--write-baseline", action="store_true", help="Write tests/baselines/*.json")
    ap.add_argument("--baseline-path", default="tests/baselines/xgb_hmumu_zh2l_metrics.json")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    schema = HMUMU_ZH2L_SCHEMA
    xgb_params = load_json(args.config)["xgb_params"]

    # Read minimal columns
    branches = list(schema.feature_names()) + [schema.label]
    if schema.weight:
        branches.append(schema.weight)
    if schema.event_id:
        branches.append(schema.event_id)

    reader = RootReader(tree_name=args.tree)
    df = reader.read(files=args.files, branches=branches)

    spec = DatasetSpec(
        features=schema.feature_names(),
        label=schema.label,
        weight=schema.weight,
        event_id=schema.event_id,
    )
    arr = build_arrays(df, spec, dtype=np.float64)

    fold_id = arr.meta["event_id"] % 4  # your fold logic (event % 4) :contentReference[oaicite:3]{index=3}

    cv = XGBoostCVClassifier(xgb_params=xgb_params, n_folds=4, store_evals=True)
    cv.fit(X=arr.X, y=arr.y, w=arr.w, fold_id=fold_id, eval_set=True, verbose=False)

    metrics = evaluate_binary_classifier(y_true=arr.y, y_score=cv.oof_pred_, sample_weight=arr.w)

    # Save artifacts
    cv.save(str(outdir / "xgb_cv_model.joblib"))
    save_json(outdir / "metrics_oof.json", metrics)

    # Optional baseline for CI
    if args.write_baseline:
        bp = Path(args.baseline_path)
        bp.parent.mkdir(parents=True, exist_ok=True)
        save_json(bp, metrics)

    print("Saved:", outdir)
    print("OOF metrics:", metrics)


if __name__ == "__main__":
    main()
