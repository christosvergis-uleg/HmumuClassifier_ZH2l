from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ATLAS_LABEL = "ATLAS-Internal"
DEFAULT_COM_ENERGY = r"$\sqrt{s}=13.6\ \mathrm{TeV}$"
DEFAULT_LUMI = r"$165\ \mathrm{fb}^{-1}$"


def _weighted_range(*arrays: np.ndarray) -> tuple[float, float] | None:
    finite_parts: list[np.ndarray] = []
    for arr in arrays:
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            finite_parts.append(x)

    if not finite_parts:
        return None

    merged = np.concatenate(finite_parts)
    lo = float(np.min(merged))
    hi = float(np.max(merged))

    if lo == hi:
        pad = 0.5 if lo == 0.0 else abs(lo) * 0.05
        lo -= pad
        hi += pad

    return lo, hi


def _normalize_hist_weights(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    n_finite = int(np.count_nonzero(finite_mask))

    if weights is None:
        if n_finite == 0:
            return None
        return np.full(n_finite, 1.0 / n_finite, dtype=float)

    w = np.asarray(weights, dtype=float)
    w = w[finite_mask]
    if w.size == 0:
        return None

    total = float(np.sum(w))
    if not np.isfinite(total) or total == 0.0:
        return None
    return w / total


def _finite_values_and_weights(
    values: np.ndarray,
    weights: np.ndarray | None,
    *,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(x)
    x = x[finite_mask]
    if x.size == 0:
        return x, None

    if weights is None:
        if not normalize:
            return x, None
        return x, np.full(x.size, 1.0 / x.size, dtype=float)

    w = np.asarray(weights, dtype=float)[finite_mask]
    if normalize:
        return x, _normalize_hist_weights(x, w)
    return x, w


def _feature_plot_cfg(
    feature_name: str,
    feature_plot_config: Mapping[str, Any] | None,
    default_bins: int,
) -> dict[str, Any]:
    cfg = dict((feature_plot_config or {}).get(feature_name, {}))

    xlim = cfg.get("xlim")
    if xlim is not None:
        if not isinstance(xlim, (list, tuple)) or len(xlim) != 2:
            raise ValueError(f"Feature '{feature_name}' xlim must be a 2-element list or tuple")
        xlim = (float(xlim[0]), float(xlim[1]))

    return {
        "bins": int(cfg.get("bins", default_bins)),
        "xlim": xlim,
        "xlabel": str(cfg.get("xlabel", feature_name)),
        "ylabel": cfg.get("ylabel"),
        "title": str(cfg.get("title", feature_name)),
    }


def _draw_atlas_label(
    fig: plt.Figure,
    *,
    atlas_label: str = DEFAULT_ATLAS_LABEL,
    com_energy: str = DEFAULT_COM_ENERGY,
    lumi: str = DEFAULT_LUMI,
) -> None:
    fig.text(0.015, 0.985, atlas_label, ha="left", va="top", fontsize=18, fontweight="bold")
    fig.text(0.015, 0.958, f"{com_energy}, {lumi}", ha="left", va="top", fontsize=15)


def plot_train_test_feature_distributions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    output_path: str | Path,
    *,
    w_train: np.ndarray | None = None,
    w_test: np.ndarray | None = None,
    bins: int = 50,
    density: bool = False,
    normalize: bool = True,
    ncols: int = 2,
    title: str | None = None,
    feature_plot_config: Mapping[str, Any] | None = None,
    atlas_label: str = DEFAULT_ATLAS_LABEL,
    com_energy: str = DEFAULT_COM_ENERGY,
    lumi: str = DEFAULT_LUMI,
    show_atlas_label: bool = True,
) -> Path:
    """Plot per-feature train/test overlays for signal/background.

    Style convention:
      - signal: red
      - background: blue
      - train: solid
      - test: dashed

    Notes:
      - normalize=True means each histogram is normalized independently to sum to 1.
      - density=True retains matplotlib's area-normalized behaviour. Prefer normalize=True
        for fixed-bin shape comparisons unless you specifically want density semantics.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(np.int64, copy=False)
    y_test = np.asarray(y_test).astype(np.int64, copy=False)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features")
    if X_train.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match number of columns in X arrays")
    if density and normalize:
        raise ValueError("Choose either density=True or normalize=True, not both")

    wtr = None if w_train is None else np.asarray(w_train, dtype=float)
    wte = None if w_test is None else np.asarray(w_test, dtype=float)

    n_features = len(feature_names)
    ncols = max(1, int(ncols))
    nrows = ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    masks = {
        "train_sig": (y_train == 1),
        "train_bkg": (y_train == 0),
        "test_sig": (y_test == 1),
        "test_bkg": (y_test == 0),
    }

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        feature_cfg = _feature_plot_cfg(feat, feature_plot_config, default_bins=bins)

        train_sig_raw = X_train[masks["train_sig"], i]
        train_bkg_raw = X_train[masks["train_bkg"], i]
        test_sig_raw = X_test[masks["test_sig"], i]
        test_bkg_raw = X_test[masks["test_bkg"], i]

        hist_range = feature_cfg["xlim"] or _weighted_range(
            train_sig_raw,
            train_bkg_raw,
            test_sig_raw,
            test_bkg_raw,
        )
        if hist_range is None:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(feature_cfg["title"])
            ax.set_axis_off()
            continue

        ws_train_sig = None if wtr is None else wtr[masks["train_sig"]]
        ws_train_bkg = None if wtr is None else wtr[masks["train_bkg"]]
        ws_test_sig = None if wte is None else wte[masks["test_sig"]]
        ws_test_bkg = None if wte is None else wte[masks["test_bkg"]]

        train_sig, ws_train_sig = _finite_values_and_weights(
            train_sig_raw, ws_train_sig, normalize=normalize
        )
        train_bkg, ws_train_bkg = _finite_values_and_weights(
            train_bkg_raw, ws_train_bkg, normalize=normalize
        )
        test_sig, ws_test_sig = _finite_values_and_weights(
            test_sig_raw, ws_test_sig, normalize=normalize
        )
        test_bkg, ws_test_bkg = _finite_values_and_weights(
            test_bkg_raw, ws_test_bkg, normalize=normalize
        )

        common_hist_kwargs = {
            "range": hist_range,
            "density": density,
            "histtype": "step",
            "linewidth": 1.6,
        }

        ax.hist(
            train_sig,
            bins=feature_cfg["bins"],
            weights=ws_train_sig,
            color="red",
            linestyle="solid",
            label="Signal train",
            **common_hist_kwargs,
        )
        ax.hist(
            test_sig,
            bins=feature_cfg["bins"],
            weights=ws_test_sig,
            color="red",
            linestyle="dashed",
            label="Signal test",
            **common_hist_kwargs,
        )
        ax.hist(
            train_bkg,
            bins=feature_cfg["bins"],
            weights=ws_train_bkg,
            color="blue",
            linestyle="solid",
            label="Background train",
            **common_hist_kwargs,
        )
        ax.hist(
            test_bkg,
            bins=feature_cfg["bins"],
            weights=ws_test_bkg,
            color="blue",
            linestyle="dashed",
            label="Background test",
            **common_hist_kwargs,
        )

        ax.set_title(feature_cfg["title"])
        ax.set_xlabel(feature_cfg["xlabel"])
        ax.set_xlim(*hist_range)

        if feature_cfg["ylabel"] is not None:
            ax.set_ylabel(str(feature_cfg["ylabel"]))
        elif density:
            ax.set_ylabel("Density")
        elif normalize:
            ax.set_ylabel("Normalized entries")
        else:
            ax.set_ylabel("Weighted entries")

        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(fontsize=9)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    if show_atlas_label:
        _draw_atlas_label(fig, atlas_label=atlas_label, com_energy=com_energy, lumi=lumi)

    if title:
        if show_atlas_label:
            fig.suptitle(title, fontsize=14, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.955))
        else:
            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        if show_atlas_label:
            fig.tight_layout(rect=(0, 0, 1, 0.955))
        else:
            fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
