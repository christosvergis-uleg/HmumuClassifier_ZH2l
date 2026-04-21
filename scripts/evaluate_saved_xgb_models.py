from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from eventclf.data import DatasetSpec, build_arrays
from eventclf.eval import evaluate_binary_classifier
from eventclf.io import RootReader
from eventclf.utils.io import ensure_dir, load_json, save_json

try:
    from eventclf.plotting import MetricsPlotter, PlotStyleConfig
    HAVE_PLOTTING = True
except Exception:
    HAVE_PLOTTING = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


DERIVED_FEATURE_DEPENDENCIES = {
    "DPHI_MET_DIMU": ["Event_MET_Phi", "Z_Phi_FSR"],
}


def _load_run_config(path: str | Path) -> dict:
    cfg = load_json(path)

    required = [
        "tree_name",
        "event_id_column",
        "n_folds_outer",
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
    df["sample_name"] = file_path.stem
    return df


def _read_unlabeled(
    reader: RootReader,
    file_path: Path,
    branches: list[str],
    sample_name: str,
) -> pd.DataFrame:
    df = reader.read(files=[str(file_path)], branches=branches)
    df["source_file"] = file_path.name
    df["sample_name"] = sample_name
    return df


def _load_training_dataframe(cfg: dict, features: list[str], weight_col: Optional[str]) -> pd.DataFrame:
    tree_name = cfg["tree_name"]
    signal_files = [Path(p) for p in cfg["signal_files"]]
    background_files = [Path(p) for p in cfg["background_files"]]
    event_id_col = cfg["event_id_column"]

    all_files = signal_files + background_files
    missing_files = [str(p) for p in all_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files from config: {missing_files}")

    branches = _branches_to_read(features, event_id_col=event_id_col, weight_col=weight_col)
    reader = RootReader(tree_name=tree_name, step_size=cfg.get("step_size", "200 MB"))

    dfs = []
    for f in signal_files:
        log.info("Reading training signal file: %s", f.name)
        dfs.append(_read_labeled(reader, f, 1, branches))
    for f in background_files:
        log.info("Reading training background file: %s", f.name)
        dfs.append(_read_labeled(reader, f, 0, branches))

    df = pd.concat(dfs, ignore_index=True)
    df = _add_derived_features(df, requested_features=features)
    return df


def _load_external_dataframe(
    cfg: dict,
    features: list[str],
    weight_col: Optional[str],
    input_file: Path,
    sample_name: str,
) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing external file: {input_file}")

    tree_name = cfg["tree_name"]
    event_id_col = cfg["event_id_column"]
    branches = _branches_to_read(features, event_id_col=event_id_col, weight_col=weight_col)

    reader = RootReader(tree_name=tree_name, step_size=cfg.get("step_size", "200 MB"))
    log.info("Reading external sample: %s", input_file.name)
    df = _read_unlabeled(reader, input_file, branches, sample_name=sample_name)
    df = _add_derived_features(df, requested_features=features)
    return df

def _save_existing_scored_dataframe(
    df: pd.DataFrame,
    outpath: Path,
    event_id_col: str,
    weight_cols: Optional[list[str]] = None,
) -> None:
    weight_cols = weight_cols or []

    cols = [c for c in [
        event_id_col,
        "label",
        "source_file",
        "sample_name",
        "weight",
        "weight_ml",
        "bdt_score",
    ] if c in df.columns]

    for c in weight_cols:
        if c in df.columns and c not in cols:
            cols.append(c)

    if outpath.suffix.lower() == ".csv":
        df[cols].to_csv(outpath, index=False)
    else:
        try:
            df[cols].to_parquet(outpath, index=False)
        except Exception:
            fallback = outpath.with_suffix(".csv")
            log.warning("Parquet save failed, falling back to CSV: %s", fallback)
            df[cols].to_csv(fallback, index=False)

def _apply_weight_mode(df: pd.DataFrame, weight_col: Optional[str], weight_mode: str, require_positive: bool) -> tuple[pd.DataFrame, str]:
    df = df.copy()

    if weight_col is None:
        df["weight"] = 1.0
        return df, "weight"

    if weight_col not in df.columns:
        raise KeyError(f"Configured weight_column '{weight_col}' not found")

    if not require_positive:
        return df, weight_col

    if weight_mode == "unit":
        df["weight_ml"] = 1.0
        return df, "weight_ml"

    if weight_mode == "abs":
        df["weight_ml"] = np.abs(df[weight_col].astype(np.float64))
        return df, "weight_ml"

    if weight_mode == "positive_only":
        mask = np.isfinite(df[weight_col]) & (df[weight_col] > 0)
        dropped = int((~mask).sum())
        df = df.loc[mask].copy()
        log.info("Dropped %d rows with non-positive/non-finite weights for ML scoring", dropped)
        return df, weight_col

    if weight_mode == "raw":
        w = df[weight_col].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weight_mode='raw' requested for ML usage, but weights are not strictly positive.")
        return df, weight_col

    raise ValueError(f"Unknown weight_mode '{weight_mode}'")


def _build_arrays_for_scoring(
    df: pd.DataFrame,
    features: list[str],
    event_id_col: str,
    label_col: Optional[str],
    weight_col: Optional[str],
) -> tuple[DatasetSpec, object]:
    spec = DatasetSpec(
        features=tuple(features),
        label=label_col if label_col is not None else "label",
        weight=weight_col,
        event_id=event_id_col,
        extra_cols=tuple(c for c in ["source_file", "sample_name"] if c in df.columns),
    )
    arr = build_arrays(df, spec, dtype=np.float64)
    return spec, arr


def _prob1(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1]
    if p.ndim == 1:
        return p
    raise ValueError(f"Unexpected predict_proba output shape: {p.shape}")


def _load_best_params(best_params_json: Path) -> list[dict]:
    payload = load_json(best_params_json)
    if "best_params_per_fold" not in payload:
        raise KeyError(f"{best_params_json} does not contain 'best_params_per_fold'")
    params = payload["best_params_per_fold"]
    if not isinstance(params, list) or len(params) == 0:
        raise ValueError("best_params_per_fold is empty or not a list")
    return params


def _load_or_fit_fold_models(
    X: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    fold_id: np.ndarray,
    best_params_per_fold: list[dict],
    model_dir: Optional[Path],
) -> list[xgb.XGBClassifier]:
    models: list[xgb.XGBClassifier] = []

    if model_dir is not None and model_dir.exists():
        model_paths = [model_dir / f"xgb_fold{k}.json" for k in range(len(best_params_per_fold))]
        if all(p.exists() for p in model_paths):
            log.info("Loading saved fold model files from %s", model_dir)
            for k, (params, path) in enumerate(zip(best_params_per_fold, model_paths)):
                model = xgb.XGBClassifier(**params)
                model.load_model(path)
                models.append(model)
            return models
        else:
            missing = [str(p) for p in model_paths if not p.exists()]
            log.warning("Not all saved fold model files were found. Will refit from saved params. Missing: %s", missing)

    log.info("Refitting fold models from saved best params (no RandomizedSearchCV).")
    for k, params in enumerate(best_params_per_fold):
        tr_idx = np.where(fold_id != k)[0]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        wtr = w[tr_idx] if w is not None else None

        model = xgb.XGBClassifier(**params)
        log.info("[Fold %d/%d] Refit model from saved params", k + 1, len(best_params_per_fold))
        if wtr is not None:
            model.fit(Xtr, ytr, sample_weight=wtr, verbose=False)
        else:
            model.fit(Xtr, ytr, verbose=False)
        models.append(model)

    return models


def _score_training_blind(
    X: np.ndarray,
    fold_id: np.ndarray,
    models: list[xgb.XGBClassifier],
) -> np.ndarray:
    n = X.shape[0]
    pred = np.full(n, np.nan, dtype=float)

    for k, model in enumerate(models):
        te_idx = np.where(fold_id == k)[0]
        log.info("[Fold %d/%d] Scoring blind subset (%d events)", k + 1, len(models), len(te_idx))
        pred[te_idx] = _prob1(model.predict_proba(X[te_idx]))

    if np.isnan(pred).any():
        raise RuntimeError("Blind predictions contain NaNs.")
    return pred


def _score_external_average(
    X: np.ndarray,
    models: list[xgb.XGBClassifier],
) -> np.ndarray:
    fold_preds = []
    for k, model in enumerate(models):
        log.info("[Fold %d/%d] Scoring external sample", k + 1, len(models))
        fold_preds.append(_prob1(model.predict_proba(X)))
    return np.mean(np.vstack(fold_preds), axis=0)


def _save_scored_dataframe(
    df: pd.DataFrame,
    event_id_col: str,
    score: np.ndarray,
    outpath: Path,
) -> None:
    out_df = df.copy()
    out_df["bdt_score"] = score

    cols = [c for c in [
        event_id_col,
        "label",
        "source_file",
        "sample_name",
        "weight",
        "weight_ml",
        "bdt_score",
    ] if c in out_df.columns]

    if outpath.suffix.lower() == ".csv":
        out_df[cols].to_csv(outpath, index=False)
    else:
        try:
            out_df[cols].to_parquet(outpath, index=False)
        except Exception:
            fallback = outpath.with_suffix(".csv")
            log.warning("Parquet save failed, falling back to CSV: %s", fallback)
            out_df[cols].to_csv(fallback, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to run config JSON")
    ap.add_argument("--best-params", default="artifacts/best_params_per_fold.json", help="Path to best_params_per_fold.json")
    ap.add_argument("--model-dir", default=None, help="Directory containing saved fold model JSON files (xgb_fold0.json, ...)")
    ap.add_argument("--mode", choices=["blind_train", "external"], required=True, help="blind_train: rescore original train sample in blind-fold mode; external: score a new file by averaging fold models")
    ap.add_argument("--input-file", default=None, help="External ROOT file to score when --mode external")
    ap.add_argument("--sample-name", default="external_sample", help="Label for external scored sample")
    ap.add_argument("--outdir", default="artifacts_eval", help="Output directory")
    ap.add_argument("--make-plots", action="store_true", help="Make plots if plotting module is available")
    ap.add_argument("--plot-lumi", type=float, default=None, help="Luminosity for plot annotation")
    ap.add_argument("--plot-status", default="Internal", help="ATLAS status text")
    args = ap.parse_args()

    cfg = _load_run_config(args.config)
    outdir = ensure_dir(args.outdir)

    features = list(cfg["features_final"])
    event_id_col = cfg["event_id_column"]
    weight_col = cfg.get("weight_column")
    weight_mode = cfg.get("weight_mode", "positive_only")
    n_folds_outer = int(cfg["n_folds_outer"])

    best_params_per_fold = _load_best_params(Path(args.best_params))
    if len(best_params_per_fold) != n_folds_outer:
        raise ValueError(
            f"Number of saved parameter sets ({len(best_params_per_fold)}) "
            f"does not match n_folds_outer ({n_folds_outer})"
        )

    # Training sample is always needed to reconstruct fold models if model files do not exist.
    train_df = _load_training_dataframe(cfg, features=features, weight_col=weight_col)
    train_df_ml, ml_weight_col = _apply_weight_mode(
        train_df,
        weight_col=weight_col,
        weight_mode=weight_mode,
        require_positive=True,
    )

    _, train_arr = _build_arrays_for_scoring(
        train_df_ml,
        features=features,
        event_id_col=event_id_col,
        label_col="label",
        weight_col=ml_weight_col,
    )

    fold_id = (train_arr.meta["event_id"].astype(np.int64) % n_folds_outer).astype(np.int64)

    model_dir = Path(args.model_dir) if args.model_dir else None
    models = _load_or_fit_fold_models(
        train_arr.X,
        train_arr.y,
        train_arr.w,
        fold_id,
        best_params_per_fold=best_params_per_fold,
        model_dir=model_dir,
    )

    if args.mode == "blind_train":
        log.info("Scoring original train/eval sample in blind-fold mode")
        blind_pred = _score_training_blind(train_arr.X, fold_id, models)
        train_df_ml_scored = train_df_ml.copy()
        train_df_ml_scored["bdt_score"] = blind_pred

        train_df_full_scored = train_df.merge(
            train_df_ml_scored[[event_id_col, "bdt_score"]],
            on=event_id_col,
            how="left",
        )
        metrics = evaluate_binary_classifier(
            y_true=train_arr.y,
            y_score=blind_pred,
            sample_weight=train_arr.w,
        )

        log.info("Blind evaluation metrics: %s", metrics)
        save_json(Path(outdir) / "metrics_blind_from_saved_models.json", metrics)
        #_save_scored_dataframe(
        #    train_df_ml,
        #    event_id_col=event_id_col,
        #    score=blind_pred,
        #    outpath=Path(outdir) / "blind_predictions_from_saved_models.parquet",
        #)
        _save_existing_scored_dataframe(
        train_df_ml_scored,
        outpath=Path(outdir) / "blind_predictions_ml.parquet",
        event_id_col=event_id_col,
        weight_cols=[ml_weight_col] if ml_weight_col is not None else [],
    )

        _save_existing_scored_dataframe(
            train_df_full_scored,
            outpath=Path(outdir) / "blind_predictions_full_signed.parquet",
            event_id_col=event_id_col,
            weight_cols=[weight_col] if weight_col is not None else [],
        )

        if args.make_plots and HAVE_PLOTTING:
            plotter = MetricsPlotter(
                PlotStyleConfig(
                    output_dir=outdir,
                    atlas_status=args.plot_status,
                    lumi_fb=args.plot_lumi,
                )
            )
            plotter.plot_roc(train_arr.y, blind_pred, train_arr.w, filename="roc_curve_saved_models.png")
            plotter.plot_prc(train_arr.y, blind_pred, train_arr.w, filename="pr_curve_saved_models.png")

            df_plot = train_df_full_scored[train_df_full_scored["bdt_score"].notna()].copy()
            signed_plot_weight_col = weight_col if (weight_col is not None and weight_col in df_plot.columns) else None
            signed_plot_weights    = df_plot[signed_plot_weight_col].to_numpy() if signed_plot_weight_col is not None else None
            plotter.plot_score_distribution(train_arr.y, blind_pred, train_arr.w, filename="score_distribution_saved_models.png", bins=80, logy=True,)
            plotter.plot_score_distribution(
                y_true=df_plot["label"].to_numpy(),
                y_score=df_plot["bdt_score"].to_numpy(),
                sample_weight=signed_plot_weights,
                filename="score_distribution_signed_weights.png",
                bins=80,
                logy=True,
            )
        elif args.make_plots:
            log.warning("Plotting requested, but eventclf.plotting is not available.")

    elif args.mode == "external":
        if args.input_file is None:
            raise ValueError("--input-file is required when --mode external")

        ext_df = _load_external_dataframe(
            cfg,
            features=features,
            weight_col=weight_col,
            input_file=Path(args.input_file),
            sample_name=args.sample_name,
        )
        ext_df["label"] = 1

        # For external scoring, do not drop rows unless you explicitly want that for ML-only scoring.
        # We keep all rows and only use weights later for plotting/evaluation choices.
        ext_weight_col = weight_col if weight_col in ext_df.columns else None

        _, ext_arr = _build_arrays_for_scoring(
            ext_df,
            features=features,
            event_id_col=event_id_col,
            label_col=None if "label" not in ext_df.columns else "label",
            weight_col=ext_weight_col,
        )

        score = _score_external_average(ext_arr.X, models)

        ext_df_scored = ext_df.copy()
        ext_df_scored["bdt_score"] = score

        _save_existing_scored_dataframe(
            ext_df_scored,
            outpath=Path(outdir) / f"scores_{args.sample_name}.parquet",
            event_id_col=event_id_col,
            weight_cols=[weight_col] if weight_col is not None else [],
        )
        log.info("Saved external scores for sample '%s'", args.sample_name)

        if "label" in ext_df.columns:
            metrics = evaluate_binary_classifier(
                y_true=ext_arr.y,
                y_score=score,
                sample_weight=ext_arr.w,
            )
            save_json(Path(outdir) / f"metrics_{args.sample_name}.json", metrics)
            log.info("External sample metrics: %s", metrics)

    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()