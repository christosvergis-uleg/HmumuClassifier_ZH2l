from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eventclf.data import DatasetSpec, build_arrays
from eventclf.eval import evaluate_binary_classifier
from eventclf.io import RootReader
from eventclf.model import XGBRotatingBlindTrainer, XGBRotatingTrainValBlindTrainer
from eventclf.utils.io import ensure_dir, load_json, save_json

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


DEFAULT_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_estimators": 800,
    "n_jobs": 8,
}

DEFAULT_PARAM_DIST = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1.0, 2.0, 5.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "gamma": [0.0, 0.1, 0.5],
    "n_estimators": [300, 600, 1000, 1500],
}

# Derived features currently supported by this training script.
# Extend this map as you move more feature engineering out of notebooks.
DERIVED_FEATURE_DEPENDENCIES = {
    "DPHI_MET_DIMU": ["Event_MET_Phi", "Z_Phi_FSR"],
}


def _load_run_config(path: str | Path) -> dict:
    cfg = load_json(path)

    required = [
        "tree_name",
        "signal_files",
        "background_files",
        "event_id_column",
        "n_folds_outer",
        "tuning",
        "features_final",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    return cfg


def _branches_to_read(features: list[str], event_id_col: str, weight_col: str | None) -> list[str]:
    branches: list[str] = []

    for feat in features:
        if feat not in DERIVED_FEATURE_DEPENDENCIES:
            branches.append(feat)

    for feat in features:
        for dep in DERIVED_FEATURE_DEPENDENCIES.get(feat, []):
            branches.append(dep)

    branches.append(event_id_col)
    if weight_col:
        branches.append(weight_col)

    # Keep deterministic order, remove duplicates.
    return list(dict.fromkeys(branches))


def _add_derived_features(df: pd.DataFrame, requested_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    if "DPHI_MET_DIMU" in requested_features and "DPHI_MET_DIMU" not in df.columns:
        needed = ["Event_MET_Phi", "Z_Phi_FSR"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(
                "Cannot build derived feature 'DPHI_MET_DIMU'. "
                f"Missing raw columns: {missing}"
            )
        dphi = np.abs(df["Event_MET_Phi"] - df["Z_Phi_FSR"])
        df["DPHI_MET_DIMU"] = np.minimum(2.0 * np.pi - dphi, dphi)

    return df


def _read_labeled(
    reader: RootReader,
    file_path: Path,
    label_value: int,
    branches: list[str],
) -> pd.DataFrame:
    df = reader.read(files=[str(file_path)], branches=branches)
    df["label"] = int(label_value)
    df["source_file"] = file_path.name
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to run config JSON")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    args = ap.parse_args()

    cfg = _load_run_config(args.config)
    outdir = ensure_dir(args.outdir)

    tree_name = cfg["tree_name"]
    signal_files = [Path(p) for p in cfg["signal_files"]]
    background_files = [Path(p) for p in cfg["background_files"]]
    features = list(cfg["features_final"])
    event_id_col = cfg["event_id_column"]
    weight_col = cfg.get("weight_column")
    weight_mode = cfg.get("weight_mode", "unit")  # one of: unit, abs, positive_only, raw

    n_folds_outer = int(cfg["n_folds_outer"])
    tuning_cfg = cfg["tuning"]

    base_params = dict(DEFAULT_BASE_PARAMS)
    base_params.update(cfg.get("base_params", {}))

    param_dist = dict(DEFAULT_PARAM_DIST)
    param_dist.update(cfg.get("param_distributions", {}))


    log.info("Loading run config from %s", args.config)
    log.info("Tree name: %s", tree_name)
    log.info("Signal files: %d | Background files: %d", len(signal_files), len(background_files))
    log.info("Number of features: %d", len(features))
    log.info("Outer folds: %d | Inner CV: %d | Tuning iterations: %d",
            n_folds_outer,
            int(tuning_cfg.get("inner_cv", max(2, n_folds_outer - 1))),
            int(tuning_cfg.get("n_iter", 100)))


    all_files = signal_files + background_files
    missing_files = [str(p) for p in all_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files from config: {missing_files}")
    log.info("All input ROOT files found (%d files)", len(all_files))

    log.info("Building branch list (including dependencies)")
    branches = _branches_to_read(features, event_id_col=event_id_col, weight_col=weight_col)
    log.info("Total branches to read: %d", len(branches))
    log.info("Reading ROOT files...")
    reader = RootReader(tree_name=tree_name, step_size=cfg.get("step_size", "200 MB"))

    log.info("Reading signal file: %s", signal_files[0].name)
    dfs = [_read_labeled(reader, signal_files[0], 1, branches)]
    for f in signal_files[1:]:
        log.info("Reading signal file: %s", f.name)
        dfs.append(_read_labeled(reader, f, 1, branches))
    for f in background_files:
        log.info("Reading background file: %s", f.name)
        dfs.append(_read_labeled(reader, f, 0, branches))

    df = pd.concat(dfs, ignore_index=True)
    log.info("Adding derived features (if needed)")
    df = _add_derived_features(df, requested_features=features)
    log.info("Combined dataframe shape: %s", df.shape)

    log.info("Checking required columns (event_id, weights, etc.)")
    if event_id_col not in df.columns:
        raise KeyError(f"Configured event_id_column '{event_id_col}' not found in loaded dataframe")


    if weight_col is None:
        df["weight"] = 1.0
        weight_col = "weight"
        log.info("No weight_column provided. Using unit weights.")
    elif weight_col not in df.columns:
        raise KeyError(f"Configured weight_column '{weight_col}' not found in loaded dataframe")

    if weight_mode == "unit":
        df["weight"] = 1.0
        weight_col = "weight"
        log.info("Using unit weights for training")

    elif weight_mode == "abs":
        df[weight_col] = np.abs(df[weight_col].astype(np.float64))
        log.info("Using absolute values of '%s' for training weights", weight_col)

    elif weight_mode == "positive_only":
        n_before = len(df)
        mask = np.isfinite(df[weight_col]) & (df[weight_col] > 0)
        df = df.loc[mask].copy()
        log.info(
            "Using only positive weights from '%s'. Dropped %d events",
            weight_col,
            n_before - len(df),
        )

    elif weight_mode == "raw":
        log.info("Using raw weights from '%s'", weight_col)

    else:
        raise ValueError(
            f"Unknown weight_mode '{weight_mode}'. "
            "Expected one of: unit, abs, positive_only, raw"
        )

    wdf = df[weight_col].to_numpy(dtype=np.float64)

    log.info(
        "Training weight column '%s': min=%.6g | max=%.6g | mean=%.6g | <=0=%d | ==0=%d | <0=%d | NaN=%d | inf=%d",
        weight_col,
        np.nanmin(wdf),
        np.nanmax(wdf),
        np.nanmean(wdf),
        int(np.sum(wdf <= 0)),
        int(np.sum(wdf == 0)),
        int(np.sum(wdf < 0)),
        int(np.sum(np.isnan(wdf))),
        int(np.sum(np.isinf(wdf))),
    )

    if np.any(~np.isfinite(wdf)):
        raise ValueError(f"Training weight column '{weight_col}' contains NaN or inf values.")

    if np.any(wdf <= 0):
        bad_rows = np.where(wdf <= 0)[0][:10]
        raise ValueError(
            f"Training weight column '{weight_col}' still contains non-positive values "
            f"after applying weight_mode='{weight_mode}'. "
            f"First bad row indices: {bad_rows.tolist()}"
        )




    spec = DatasetSpec(
        features=tuple(features),
        label="label",
        weight=weight_col,
        event_id=event_id_col,
        extra_cols=("source_file",),
    )
    log.info("Building training arrays (X, y, weights)")
    log.info("Final feature set: %s", features)
    arr = build_arrays(df, spec, dtype=np.float64)
    log.info("Array shapes: X=%s | y=%s", arr.X.shape, arr.y.shape)

    log.info("Assigning folds using %s %% %d", event_id_col, n_folds_outer)
    fold_id = (arr.meta["event_id"].astype(np.int64) % n_folds_outer).astype(np.int64)
    unique, counts = np.unique(fold_id, return_counts=True)
    fold_summary = {int(k): int(v) for k, v in zip(unique, counts)}
    log.info("Fold distribution: %s", fold_summary)

    trainer = XGBRotatingBlindTrainer(
        base_params=base_params,
        param_distributions=param_dist,
        n_folds=n_folds_outer,
        inner_cv=int(tuning_cfg.get("inner_cv", max(2, n_folds_outer - 1))),
        n_iter=int(tuning_cfg.get("n_iter", 100)),
        scoring=tuning_cfg.get("scoring", "neg_log_loss"),
        random_state=int(cfg.get("random_state", 42)),
        verbose=int(cfg.get("verbose", 1)),
    )

    '''
    # Simple rotation
    trainer = XGBRotatingTrainValBlindTrainer(
        base_params=base_params,
        param_distributions=param_dist,
        n_folds=n_folds_outer,
        n_iter=int(tuning_cfg.get("n_iter", 20)),
        scoring=tuning_cfg.get("scoring", "average_precision"),
        random_state=int(cfg.get("random_state", 42)),
        verbose=int(cfg.get("verbose", 0)),
        refit_on_train_val=False,
    )
    '''
    plot_cfg = cfg.get("feature_plotting", {})
    feature_plot_dir = outdir / plot_cfg.get("subdir", "feature_plots") if plot_cfg.get("enabled", False) else None
    atlas_cfg = plot_cfg.get("atlas_label", {})


    log.info("Starting XGBoost rotating blind training")
    log.info("Base params: %s", base_params)
    log.info("Parameter search space size: %d", len(param_dist))
    trainer.fit(
        arr.X,
        arr.y,
        arr.w,
        fold_id=fold_id,
        feature_names=features,
        feature_plot_dir=feature_plot_dir,
        feature_plot_bins=int(plot_cfg.get("bins", 50)),
        feature_plot_density=bool(plot_cfg.get("density", False)),
        feature_plot_normalize=bool(plot_cfg.get("normalize", not plot_cfg.get("density", False))),
        feature_plot_ncols=int(plot_cfg.get("ncols", 2)),
        feature_plot_config=plot_cfg.get("per_feature", {}),
        atlas_label=str(atlas_cfg.get("label", "ATLAS-Internal")),
        com_energy=str(atlas_cfg.get("com_energy", r"$\sqrt{s}=13.6\ \mathrm{TeV}$")),
        lumi=str(atlas_cfg.get("lumi", r"$165\ \mathrm{fb}^{-1}$")),
        show_atlas_label=bool(atlas_cfg.get("enabled", True)),
    )
    log.info("Training complete")

    df["BDT_score_blind"] = trainer.blind_pred_

    log.info("Evaluating blind predictions")
    metrics = evaluate_binary_classifier(
        y_true=arr.y,
        y_score=trainer.blind_pred_,
        sample_weight=arr.w,
    )
    log.info("Metrics: %s", metrics)

    log.info("Saving outputs to %s", outdir)
    save_json(outdir / "metrics_blind.json", metrics)
    save_json(outdir / "best_params_per_fold.json", {"best_params_per_fold": trainer.best_params_per_fold_})

    from eventclf.plotting.metrics import MetricsPlotter, PlotStyleConfig

    plotter = MetricsPlotter(
        PlotStyleConfig(
            output_dir=outdir,
            atlas_status="Internal",
            lumi_fb=165.0,
            sqrts_tev=13.6,
            extra_text=r"$ZH \to \nu\nu + \mu\mu$",
            title_fontsize=18,
            label_fontsize=16,
            tick_fontsize=14,
            legend_fontsize=13,
            atlas_fontsize=15,
            legend_loc="best",
        )
    )

    plotter.plot_roc(
        y_true=arr.y,
        y_score=trainer.blind_pred_,
        sample_weight=arr.w,
        filename="roc_curve.png",
    )

    plotter.plot_prc(
        y_true=arr.y,
        y_score=trainer.blind_pred_,
        sample_weight=arr.w,
        filename="pr_curve.png",
    )

    plotter.plot_score_distribution(
        y_true=arr.y,
        y_score=trainer.blind_pred_,
        sample_weight=arr.w,
        filename="score_distribution.png",
        bins=80,
        logy=True,
    )

    plotter.plot_sig_eff_vs_bkg_eff(
        y_true=arr.y,
        y_score=trainer.blind_pred_,
        sample_weight=arr.w,
        filename="sigeff_vs_bkgrej.png",
    )

    plotter.plot_confusion_matrix(
        y_true=arr.y,
        y_score=trainer.blind_pred_,
        sample_weight=arr.w,
        threshold=0.5,
        filename="confusion_matrix.png",
        normalize="true",
    )


    print("Saved outputs to:", outdir)
    print("Features used:", features)
    print("Blind metrics:", metrics)


if __name__ == "__main__":
    main()
