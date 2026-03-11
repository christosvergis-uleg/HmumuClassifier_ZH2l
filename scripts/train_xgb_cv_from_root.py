from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from eventclf.config import HMUMU_ZH2L_SCHEMA
from eventclf.io import RootReader
from eventclf.data import DatasetSpec, build_arrays
from eventclf.model import XGBRotatingBlindTrainer
from eventclf.eval import evaluate_binary_classifier
from eventclf.utils.io import ensure_dir, save_json


# Your original setting
FEATURES = [
    "Muons_PT_Lead", "Muons_PT_Sub", "Event_VT_over_HT", "dR_mu0_mu1",
    "Jets_jetMultip", "Event_MET", "Muons_CosThetaStar", "Event_MET_Sig",# "DPHI_MET_DIMU"
]

# Pretty labels: keep for plotting scripts (not needed for training itself)
LABELS = {
    "Muons_PT_Lead": r"$p_{T}^{\mu_1}$",
    "Muons_PT_Sub": r"$p_{T}^{\mu_2}$",
    "Event_VT_over_HT": r"$V_{T}/H_{T}$",
    "DPHI_MET_DIMU": r"$\Delta\phi (\mu\mu, E_{T}^{miss})$",
    "dR_mu0_mu1": r"$\Delta R(\mu_1, \mu_2)$",
    "Jets_jetMultip": r"$N_{j}$",
    "Event_MET": r"$E_{T}^{miss}$",
    "Muons_CosThetaStar": r"$cos \theta^{*}$",
    "Event_MET_Sig": r"$\sigma(E_{T}^{miss})$",
}


def read_labeled(
    reader: RootReader,
    file_path: Path,
    label_value: int,
    branches: list[str],
    tree_name: str,
) -> pd.DataFrame:
    df = reader.read(files=[str(file_path)], branches=branches)
    df["label"] = int(label_value)
    df["source_file"] = file_path.name
    return df


def main() -> None:
    data_dir = Path("data")
    outdir = ensure_dir("artifacts")

    # Adjust if your TTree name differs
    TREE_NAME = "tree_Hmumu"

    # Identify files
    sig_file = data_dir / "signal_116m133.root"
    bkg_files = [
        data_dir / "diboson_116m133.root",
        data_dir / "dy_116m133.root",
        data_dir / "TOP_116m133.root",
        #data_dir / "signal_other_116m133.root",  # treated as background per your request
    ]

    missing = [p for p in [sig_file, *bkg_files] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    # Schema-driven branches
    schema = HMUMU_ZH2L_SCHEMA

    # We try to read event_id if present; if it isn't, we will fallback later.
    branches = list(FEATURES)
    if schema.event_id:
        branches.append(schema.event_id)
    if schema.weight:
        branches.append(schema.weight)

    reader = RootReader(tree_name=TREE_NAME, step_size="200 MB")

    # Load signal + background, create label column
    dfs = [read_labeled(reader, sig_file, 1, branches, TREE_NAME)]
    for f in bkg_files:
        dfs.append(read_labeled(reader, f, 0, branches, TREE_NAME))

    df = pd.concat(dfs, ignore_index=True)

    # If event_id column missing (common when tree name differs / branch differs), fallback
    if schema.event_id and schema.event_id not in df.columns:
        df["event"] = np.arange(len(df), dtype=np.int64)
        event_id_col = "event"
    else:
        event_id_col = schema.event_id or "event"

    # If no weights, use 1.0 (you can replace later with real event weights)
    if schema.weight and schema.weight in df.columns:
        weight_col = schema.weight
    else:
        df["weight"] = 1.0
        weight_col = "weight"

    # Build arrays
    spec = DatasetSpec(
        features=tuple(FEATURES),
        label="label",
        weight=weight_col,
        event_id=event_id_col,
        extra_cols=("source_file",),
    )

    arr = build_arrays(df, spec, dtype=np.float64)

    fold_id = (arr.meta["event_id"] % 4).astype(np.int64)
    # Make sure your df has: FEATURES + label + weight + event
    fold_id = (df["event"].to_numpy(dtype=np.int64) % 5)

    spec = DatasetSpec(
        features=tuple(FEATURES),
        label="label",
        weight="weight",     # or your real weight column
        event_id="event",
    )

    arr = build_arrays(df, spec, dtype=np.float64)

    # Base params (fixed stuff)
    base_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=800,    # baseline, tuning can override
        n_jobs=8,
    )

    # Parameter distributions for RandomSearch
    # Keep lists; RandomizedSearchCV samples from them.
    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1.0, 2.0, 5.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "gamma": [0.0, 0.1, 0.5],
        "n_estimators": [300, 600, 1000, 1500],
    }

    trainer = XGBRotatingBlindTrainer(
        base_params=base_params,
        param_distributions=param_dist,
        n_folds=5,
        inner_cv=4,
        n_iter=100,
        scoring="neg_log_loss",
        random_state=42,
        verbose=1,
    )

    trainer.fit(arr.X, arr.y, arr.w, fold_id=fold_id)

    df["BDT_score_blind"] = trainer.blind_pred_

    metrics = evaluate_binary_classifier(y_true=arr.y, y_score=trainer.blind_pred_, sample_weight=arr.w)
    save_json("artifacts/metrics_blind.json", metrics)
    save_json("artifacts/best_params_per_fold.json", {"best_params_per_fold": trainer.best_params_per_fold_})
    print("Blind metrics:", metrics)


if __name__ == "__main__":
    main()
