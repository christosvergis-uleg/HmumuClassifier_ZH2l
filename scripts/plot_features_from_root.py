from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eventclf.io import RootReader
from eventclf.monitoring.feature_checks import feature_summary
from eventclf.utils.io import ensure_dir, load_json, save_json

DERIVED_FEATURE_DEPENDENCIES = {
    "DPHI_MET_DIMU": ["Event_MET_Phi", "Z_Phi_FSR"],
}


def _load_run_config(path: str | Path) -> dict:
    cfg = load_json(path)
    required = ["tree_name", "signal_files", "background_files", "features_final"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    return cfg


def _branches_to_read(features: list[str], weight_col: str | None) -> list[str]:
    branches: list[str] = []
    for feat in features:
        if feat not in DERIVED_FEATURE_DEPENDENCIES:
            branches.append(feat)
    for feat in features:
        for dep in DERIVED_FEATURE_DEPENDENCIES.get(feat, []):
            branches.append(dep)
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


def _read_labeled(reader: RootReader, file_path: Path, label_value: int, branches: list[str]) -> pd.DataFrame:
    df = reader.read(files=[str(file_path)], branches=branches)
    df["label"] = int(label_value)
    df["source_file"] = file_path.name
    return df


def _weighted_quantile(values: np.ndarray, quantiles: Iterable[float], weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != values.shape:
        raise ValueError("weights must have the same shape as values")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def _feature_range(xs: np.ndarray, xb: np.ndarray, ws: np.ndarray | None, wb: np.ndarray | None, qlo: float, qhi: float) -> tuple[float, float]:
    vals = []
    for x, w in ((xs, ws), (xb, wb)):
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x)
        x = x[mask]
        w_local = None
        if w is not None:
            w = np.asarray(w, dtype=float)
            w_local = w[mask]
        if x.size:
            vals.append(_weighted_quantile(x, [qlo, qhi], w_local))
    if not vals:
        return 0.0, 1.0
    bounds = np.asarray(vals, dtype=float)
    lo = float(np.nanmin(bounds[:, 0]))
    hi = float(np.nanmax(bounds[:, 1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if lo == hi:
        pad = 0.5 if lo == 0.0 else 0.05 * abs(lo)
        return lo - pad, hi + pad
    return lo, hi


def _normalize_weights(weights: np.ndarray | None, size: int) -> np.ndarray | None:
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    if weights.size != size:
        raise ValueError("weights must match histogram values")
    total = np.sum(weights)
    if total <= 0.0 or not np.isfinite(total):
        return None
    return weights / total


def plot_feature_grid(df: pd.DataFrame, features: list[str], label_col: str, weight_col: str | None, outpath: Path, bins: int, density: bool, quantile_range: tuple[float, float] = (0.01, 0.99)) -> dict[str, dict]:
    sig = df[df[label_col] == 1].copy()
    bkg = df[df[label_col] == 0].copy()
    ws_all = sig[weight_col].to_numpy(dtype=float) if weight_col else None
    wb_all = bkg[weight_col].to_numpy(dtype=float) if weight_col else None
    n = len(features)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    summary: dict[str, dict] = {}
    for ax, feat in zip(axes, features):
        xs = sig[feat].to_numpy(dtype=float)
        xb = bkg[feat].to_numpy(dtype=float)
        sig_mask = np.isfinite(xs)
        bkg_mask = np.isfinite(xb)
        xs = xs[sig_mask]
        xb = xb[bkg_mask]
        ws = None if ws_all is None else ws_all[sig_mask]
        wb = None if wb_all is None else wb_all[bkg_mask]
        hist_ws = _normalize_weights(ws, xs.size) if density else ws
        hist_wb = _normalize_weights(wb, xb.size) if density else wb
        lo, hi = _feature_range(xs, xb, ws, wb, qlo=quantile_range[0], qhi=quantile_range[1])
        bin_edges = np.linspace(lo, hi, bins + 1)
        ax.hist(xs, bins=bin_edges, weights=hist_ws, histtype="step", linewidth=1.6, label="Signal")
        ax.hist(xb, bins=bin_edges, weights=hist_wb, histtype="step", linewidth=1.6, label="Background")
        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("Normalized entries" if density else "Weighted entries")
        ax.legend()
        ax.grid(alpha=0.25)
        feat_summary = {"signal": feature_summary(xs), "background": feature_summary(xb)}
        if ws is not None and ws.size:
            feat_summary["signal"]["sumw"] = float(np.sum(ws))
        if wb is not None and wb.size:
            feat_summary["background"]["sumw"] = float(np.sum(wb))
        summary[feat] = feat_summary
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot input feature distributions from ROOT files")
    ap.add_argument("--config", required=True, help="Path to run config JSON")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    ap.add_argument("--output-name", default="input_feature_distributions.png", help="Plot filename")
    ap.add_argument("--bins", type=int, default=50, help="Number of bins per feature")
    ap.add_argument("--density", action="store_true", help="Normalize each class histogram to unit area before plotting")
    args = ap.parse_args()

    cfg = _load_run_config(args.config)
    outdir = ensure_dir(args.outdir)
    tree_name = cfg["tree_name"]
    signal_files = [Path(p) for p in cfg["signal_files"]]
    background_files = [Path(p) for p in cfg["background_files"]]
    features = list(cfg["features_final"])
    weight_col = cfg.get("weight_column")
    all_files = signal_files + background_files
    missing_files = [str(p) for p in all_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files from config: {missing_files}")
    branches = _branches_to_read(features, weight_col=weight_col)
    reader = RootReader(tree_name=tree_name, step_size=cfg.get("step_size", "200 MB"))
    dfs = [_read_labeled(reader, signal_files[0], 1, branches)]
    for f in signal_files[1:]:
        dfs.append(_read_labeled(reader, f, 1, branches))
    for f in background_files:
        dfs.append(_read_labeled(reader, f, 0, branches))
    df = pd.concat(dfs, ignore_index=True)
    df = _add_derived_features(df, requested_features=features)
    if weight_col is None:
        df["weight"] = 1.0
        weight_col = "weight"
    elif weight_col not in df.columns:
        raise KeyError(f"Configured weight_column '{weight_col}' not found in loaded dataframe")
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        raise KeyError(f"Configured features not found after loading/derivation: {missing_features}")
    plot_path = outdir / args.output_name
    summary_path = outdir / "input_feature_summary.json"
    summary = plot_feature_grid(df=df, features=features, label_col="label", weight_col=weight_col, outpath=plot_path, bins=args.bins, density=bool(args.density))
    save_json(summary_path, summary)
    print("Saved feature plot to:", plot_path)
    print("Saved feature summary to:", summary_path)
    print("Features plotted:", features)


if __name__ == "__main__":
    main()
