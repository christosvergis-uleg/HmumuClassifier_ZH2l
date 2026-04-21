from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class PlotStyleConfig:
    figsize: tuple[float, float] = (8.0, 6.0)
    dpi: int = 160

    title_fontsize: int = 18
    label_fontsize: int = 16
    tick_fontsize: int = 14
    legend_fontsize: int = 13
    atlas_fontsize: int = 16
    text_fontsize: int = 14

    legend_loc: str = "best"
    grid: bool = True
    tight_layout: bool = True

    atlas_status: str = "Internal"
    lumi_fb: Optional[float] = None
    sqrts_tev: float = 13.6
    extra_text: Optional[str] = None

    output_dir: str | Path = "artifacts"


class MetricsPlotter:
    """
    Plotting helper for binary classifier evaluation.

    Intended usage:
      - ROC
      - PRC
      - score distributions
      - confusion matrix

    Tunable styling includes ATLAS-style text, fonts, legend placement, etc.
    """

    def __init__(self, style: Optional[PlotStyleConfig] = None) -> None:
        self.style = style or PlotStyleConfig()
        self.output_dir = Path(self.style.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_roc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        title: str = "ROC curve",
        filename: str = "roc_curve.png",
        curve_label: Optional[str] = None,
    ) -> Path:
        y_true, y_score, sample_weight = self._validate_inputs(y_true, y_score, sample_weight)

        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
        roc_auc = auc(fpr, tpr)

        fig, ax = self._make_figure()
        label = curve_label or f"ROC (AUC = {roc_auc:.4f})"

        ax.plot(fpr, tpr, linewidth=2.2, label=label)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.5, label="Random")

        ax.set_xlabel("False positive rate", fontsize=self.style.label_fontsize)
        ax.set_ylabel("True positive rate", fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.legend(loc=self.style.legend_loc, fontsize=self.style.legend_fontsize)
        self._style_axes(ax)
        self._add_atlas_label(ax)

        return self._save(fig, filename)

    def plot_prc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        title: str = "Precision-Recall curve",
        filename: str = "pr_curve.png",
        curve_label: Optional[str] = None,
        show_baseline: bool = True,
    ) -> Path:
        y_true, y_score, sample_weight = self._validate_inputs(y_true, y_score, sample_weight)

        precision, recall, _ = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight
        )
        ap = average_precision_score(y_true, y_score, sample_weight=sample_weight)
        pos_rate = self._positive_rate(y_true, sample_weight)

        fig, ax = self._make_figure()
        label = curve_label or f"PRC (AP = {ap:.4f})"

        ax.plot(recall, precision, linewidth=2.2, label=label)

        if show_baseline:
            ax.axhline(pos_rate, linestyle="--", linewidth=1.5, label=f"Baseline = {pos_rate:.4e}")

        ax.set_xlabel("Recall", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Precision", fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.set_yscale("log")
        ax.set_ylim(max(1e-4, pos_rate / 2), 1.05)

        ax.legend(loc=self.style.legend_loc, fontsize=self.style.legend_fontsize)
        self._style_axes(ax)
        self._add_atlas_label(ax)

        return self._save(fig, filename)

    def plot_score_distribution(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        title: str = "Score distribution",
        filename: str = "score_distribution.png",
        bins: int = 60,
        density: bool = True,
        signal_label: str = "Signal",
        background_label: str = "Background",
        logy: bool = False,
    ) -> Path:
        y_true, y_score, sample_weight = self._validate_inputs(y_true, y_score, sample_weight)

        sig_mask = (y_true == 1)
        bkg_mask = (y_true == 0)

        sig_scores = y_score[sig_mask]
        bkg_scores = y_score[bkg_mask]

        sig_weights = sample_weight[sig_mask] if sample_weight is not None else None
        bkg_weights = sample_weight[bkg_mask] if sample_weight is not None else None

        fig, ax = self._make_figure()

        ax.hist(
            bkg_scores,
            bins=bins,
            weights=bkg_weights,
            density=density,
            histtype="step",
            linewidth=2.0,
            label=background_label,
        )
        ax.hist(
            sig_scores,
            bins=bins,
            weights=sig_weights,
            density=density,
            histtype="step",
            linewidth=2.0,
            label=signal_label,
        )

        ax.set_xlabel("Model score", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Density" if density else "Events", fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.legend(loc=self.style.legend_loc, fontsize=self.style.legend_fontsize)

        if logy:
            ax.set_yscale("log")

        self._style_axes(ax)
        self._add_atlas_label(ax)

        return self._save(fig, filename)
    
    def plot_sig_eff_vs_bkg_eff(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        title: str = "Signal efficiency vs background efficiency",
        filename: str = "sigeff_vs_bkgeff.png",
        logy: bool = True,
    ) -> Path:
        y_true, y_score, sample_weight = self._validate_inputs(y_true, y_score, sample_weight)

        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)

        fig, ax = self._make_figure()
        bkg_rej = 1.0 / fpr  

        ax.plot(tpr, bkg_rej, linewidth=2.2)

        ax.set_xlabel("Signal efficiency", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Background rejection", fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)

        if logy:
            ax.set_yscale("log")

        self._style_axes(ax)
        self._add_atlas_label(ax)

        return self._save(fig, filename)

    def plot_score_distribution_multi(
        self,
        backgrounds: list[tuple[str, np.ndarray, Optional[np.ndarray]]],
        signals: list[tuple[str, np.ndarray, Optional[np.ndarray]]],
        title: str = "Score distribution",
        filename: str = "score_distribution_multi.png",
        bins: int = 80,
        density: bool = True,
        logy: bool = False,
    ) -> Path:
        fig, ax = self._make_figure()

        for label, scores, weights in backgrounds:
            ax.hist(
                scores,
                bins=bins,
                weights=weights,
                density=density,
                histtype="step",
                linewidth=2.0,
                label=label,
            )

        for label, scores, weights in signals:
            ax.hist(
                scores,
                bins=bins,
                weights=weights,
                density=density,
                histtype="step",
                linewidth=2.0,
                label=label,
            )

        ax.set_xlabel("Model score", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Density" if density else "Events", fontsize=self.style.label_fontsize)
        ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.legend(loc=self.style.legend_loc, fontsize=self.style.legend_fontsize)

        if logy:
            ax.set_yscale("log")

        self._style_axes(ax)
        self._add_atlas_label(ax)

        return self._save(fig, filename)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.5,
        sample_weight: Optional[np.ndarray] = None,
        title: str = "Confusion matrix",
        filename: str = "confusion_matrix.png",
        normalize: Optional[str] = None,
        class_labels: Sequence[str] = ("Background", "Signal"),
    ) -> Path:
        y_true, y_score, sample_weight = self._validate_inputs(y_true, y_score, sample_weight)

        y_pred = (y_score >= threshold).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            normalize=normalize,
        )

        fig, ax = self._make_figure()
        im = ax.imshow(cm, aspect="auto")

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=self.style.tick_fontsize)

        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=self.style.tick_fontsize)
        ax.set_yticklabels(class_labels, fontsize=self.style.tick_fontsize)

        ax.set_xlabel("Predicted label", fontsize=self.style.label_fontsize)
        ax.set_ylabel("True label", fontsize=self.style.label_fontsize)
        ax.set_title(f"{title} (threshold = {threshold:.3f})", fontsize=self.style.title_fontsize)

        fmt = ".3f" if normalize is not None else ".0f"
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=self.style.text_fontsize,
                )

        self._style_axes(ax, grid=False)
        self._add_atlas_label(ax)

        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_figure(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.style.figsize, dpi=self.style.dpi)
        return fig, ax

    def _style_axes(self, ax: plt.Axes, grid: Optional[bool] = None) -> None:
        use_grid = self.style.grid if grid is None else grid

        ax.tick_params(axis="both", labelsize=self.style.tick_fontsize)
        if use_grid:
            ax.grid(True, alpha=0.3)

    def _add_atlas_label(self, ax: plt.Axes) -> None:
        atlas_line = rf"$\bf{{ATLAS}}$ {self.style.atlas_status}"

        info_parts: list[str] = []
        if self.style.lumi_fb is not None:
            info_parts.append(rf"$L = {self.style.lumi_fb:.1f}\ \mathrm{{fb}}^{{-1}}$")
        info_parts.append(rf"$\sqrt{{s}} = {self.style.sqrts_tev:.1f}\ \mathrm{{TeV}}$")

        info_line = ", ".join(info_parts)

        lines = [atlas_line, info_line]
        if self.style.extra_text:
            lines.append(self.style.extra_text)

        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=self.style.atlas_fontsize,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.6),
        )

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        outpath = self.output_dir / filename
        if self.style.tight_layout:
            fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        return outpath

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_score: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        y_true = np.asarray(y_true).astype(int, copy=False)
        y_score = np.asarray(y_score).astype(float, copy=False)

        if y_true.ndim != 1:
            raise ValueError("y_true must be 1D")
        if y_score.ndim != 1:
            raise ValueError("y_score must be 1D")
        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).astype(float, copy=False)
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1D")
            if len(sample_weight) != len(y_true):
                raise ValueError("sample_weight must have the same length as y_true")

        return y_true, y_score, sample_weight

    @staticmethod
    def _positive_rate(y_true: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
        if sample_weight is None:
            return float(np.mean(y_true == 1))

        wsum = np.sum(sample_weight)
        if wsum <= 0.0:
            raise ValueError("Sum of sample weights must be positive")

        return float(np.sum(sample_weight[y_true == 1]) / wsum)