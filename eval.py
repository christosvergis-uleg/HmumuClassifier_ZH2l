import pandas as pd
from eventclf.plotting import MetricsPlotter, PlotStyleConfig
import numpy as np
import matplotlib.pyplot as plt


df_main = pd.read_parquet("artifacts_eval_xgbrot_niter100_ap/blind_predictions_full_signed.parquet")
df_other = pd.read_parquet("artifacts_eval_xgbrot_niter100_ap/scores_signal_other.parquet")
#df_other["weight"]=0
# Keep only rows that actually have scores
df_main = df_main[df_main["bdt_score"].notna()].copy()
df_other = df_other[df_other["bdt_score"].notna()].copy()

# Main sample split
df_bkg = df_main[df_main["label"] == 0].copy()
df_sig_main = df_main[df_main["label"] == 1].copy()

plotter = MetricsPlotter(
    PlotStyleConfig(
        output_dir="artifacts_eval",
        atlas_status="Internal",
        lumi_fb=165.0,
        sqrts_tev=13.6,
        extra_text=r"$ZH \rightarrow \nu\nu + \mu\mu$",
    )
)

plotter.plot_score_distribution_multi(
    backgrounds=[
        (
            "Background",
            df_bkg["bdt_score"].to_numpy(),
            df_bkg["weight"].to_numpy(),
        ),
    ],
    signals=[
        (
            "Signal (train)",
            df_sig_main["bdt_score"].to_numpy(),
            df_sig_main["weight"].to_numpy(),
        ),
        (
            "Signal (other)",
            df_other["bdt_score"].to_numpy(),
            df_other["weight"].to_numpy(),
        ),
    ],
    filename="score_distribution_with_signal_other.png",
    bins=80,
    logy=True,
)

"""
print(df_bkg.head(5))

print(df_sig_main.head(5))

print(df_other.head(5))
df_all = pd.concat([df_sig_main, df_other, df_bkg], ignore_index=True)
def asimov_significance(s, b):
    if b <= 0 or s <= 0:
        return 0.0
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))

df_sig = df_all[df_all["label"] == 1]
df_bkg = df_all[df_all["label"] == 0]

a_vals = np.linspace(0.0, 0.9, 91)
b_vals = np.linspace(0.1, 1.0, 91)

#print(a_vals)

#exit(0)
results = []

for a in a_vals:
    print("a=",a)
    for b in b_vals:
        if b <= a:
            continue

        # Region R1: [a, b]
        sig_r1 = df_sig[(df_sig.bdt_score >= a) & (df_sig.bdt_score < b)]["weight"].sum()
        bkg_r1 = df_bkg[(df_bkg.bdt_score >= a) & (df_bkg.bdt_score < b)]["weight"].sum()

        # Region R2: [b, 1]
        sig_r2 = df_sig[df_sig.bdt_score >= b]["weight"].sum()
        bkg_r2 = df_bkg[df_bkg.bdt_score >= b]["weight"].sum()

        Z1 = asimov_significance(sig_r1, bkg_r1)
        Z2 = asimov_significance(sig_r2, bkg_r2)

        Ztot = np.sqrt(Z1**2 + Z2**2)

        results.append((a, b, Z1, Z2, Ztot))

df_scan = pd.DataFrame(results, columns=["a", "b", "Z1", "Z2", "Ztot"])

import matplotlib.pyplot as plt

pivot = df_scan.pivot(index="a", columns="b", values="Ztot")

plt.figure()
plt.imshow(pivot, origin="lower", aspect="auto")
plt.colorbar(label="Z_total")

plt.xlabel("b")
plt.ylabel("a")
plt.title("Z_total scan over (a, b)")

plt.show()

best = df_scan.loc[df_scan["Ztot"].idxmax()]
print(best)
"""
# ============================================================
# 1. Prepare inputs
# ============================================================
df_sig_main = df_sig_main.copy()
df_other = df_other.copy()
df_bkg = df_bkg.copy()

# Safer than relying on sample_name
df_sig_main["group"] = "signal_main"
df_other["group"] = "signal_other"
df_bkg["group"] = "background"

df_all = pd.concat([df_sig_main, df_other, df_bkg], ignore_index=True)
df_sigall = pd.concat([df_sig_main, df_other], ignore_index=True)

scores = df_all["bdt_score"].to_numpy()
weights = df_all["weight"].to_numpy()
groups = df_all["group"].to_numpy()

mask_sig_main = (groups == "signal_main")
mask_sig_other = (groups == "signal_other")
mask_bkg = (groups == "background")

# ============================================================
# 2. Asimov significance
# ============================================================
def asimov_array(s, b):
    """
    Vectorized Asimov significance:
        Z = sqrt(2 * [ (s+b) ln(1+s/b) - s ])
    Returns 0 when s<=0 or b<=0.
    """
    out = np.zeros_like(s, dtype=float)
    mask = (s > 0) & (b > 0)
    out[mask] = np.sqrt(
        2.0 * ((s[mask] + b[mask]) * np.log1p(s[mask] / b[mask]) - s[mask])
    )
    return out


# ============================================================
# 3. Threshold grid
# ============================================================
n_thresholds = 401
t = np.linspace(0.0, 1.0, n_thresholds)

# Threshold values used for cuts
thr = t[:-1]
n = len(thr)

# ============================================================
# 4. Histogram weighted yields by score bin
# ============================================================
hist_sig_main, _ = np.histogram(
    scores, bins=t, weights=weights * mask_sig_main
)
hist_sig_other, _ = np.histogram(
    scores, bins=t, weights=weights * mask_sig_other
)
hist_bkg, _ = np.histogram(
    scores, bins=t, weights=weights * mask_bkg
)

# ============================================================
# 5. Cumulative yields for score >= threshold
# ============================================================
sig_main_ge = np.cumsum(hist_sig_main[::-1])[::-1]
sig_other_ge = np.cumsum(hist_sig_other[::-1])[::-1]
bkg_ge = np.cumsum(hist_bkg[::-1])[::-1]

high_cut = 0.925

print("Background yield above 0.925:",
      df_bkg.loc[df_bkg["bdt_score"] >= high_cut, "weight"].sum())

print("Signal yield above 0.925:",
      df_sigall.loc[df_sigall["bdt_score"] >= high_cut, "weight"].sum())


sig_full_ge = sig_main_ge + sig_other_ge

# ============================================================
# 6. Build pairwise yields for R1=[a,b), R2=[b,1]
# ============================================================
Smain_a = sig_main_ge[:, None]
Smain_b = sig_main_ge[None, :]

Sother_a = sig_other_ge[:, None]
Sother_b = sig_other_ge[None, :]

Sfull_a = sig_full_ge[:, None]
Sfull_b = sig_full_ge[None, :]

B_a = bkg_ge[:, None]
B_b = bkg_ge[None, :]

# Main-only yields, for optimization
s_main_r1 = Smain_a - Smain_b
s_main_r2 = np.broadcast_to(Smain_b, (n, n))

# Other signal yields
s_other_r1 = Sother_a - Sother_b
s_other_r2 = np.broadcast_to(Sother_b, (n, n))

# Full signal yields, for final evaluation
s_full_r1 = Sfull_a - Sfull_b
s_full_r2 = np.broadcast_to(Sfull_b, (n, n))

# Background yields
bkg_r1 = B_a - B_b
bkg_r2 = np.broadcast_to(B_b, (n, n))

# ============================================================
# 7. Optimize using signal_main only
# ============================================================
Z1_opt = asimov_array(s_main_r1, bkg_r1)
Z2_opt = asimov_array(s_main_r2, bkg_r2)
Ztot_opt = np.sqrt(Z1_opt**2 + Z2_opt**2)

valid = thr[:, None] < thr[None, :]
Ztot_opt_masked = np.where(valid, Ztot_opt, np.nan)

imax = np.nanargmax(Ztot_opt_masked)
i_best, j_best = np.unravel_index(imax, Ztot_opt_masked.shape)

a_best = thr[i_best]
b_best = thr[j_best]

# ============================================================
# 8. Evaluate final reported Z using full signal
# ============================================================
Z1_full = asimov_array(s_full_r1, bkg_r1)
Z2_full = asimov_array(s_full_r2, bkg_r2)
Ztot_full = np.sqrt(Z1_full**2 + Z2_full**2)

# ============================================================
# 9. Print results
# ============================================================
print("=" * 70)
print("Best thresholds found by optimizing ONLY signal_main")
print("=" * 70)
print(f"a_best = {a_best:.6f}")
print(f"b_best = {b_best:.6f}")
print()

print("Optimization objective (signal_main only):")
print(f"Z1_opt(main only)     = {Z1_opt[i_best, j_best]:.6f}")
print(f"Z2_opt(main only)     = {Z2_opt[i_best, j_best]:.6f}")
print(f"Ztot_opt(main only)   = {Ztot_opt[i_best, j_best]:.6f}")
print()

print("Final reported score (signal_main + signal_other):")
print(f"Z1_full               = {Z1_full[i_best, j_best]:.6f}")
print(f"Z2_full               = {Z2_full[i_best, j_best]:.6f}")
print(f"Ztot_full             = {Ztot_full[i_best, j_best]:.6f}")
print()

print("=" * 70)
print(f"Yields in R1 = [{a_best:.6f}, {b_best:.6f})")
print("=" * 70)
print(f"signal_main           = {s_main_r1[i_best, j_best]:.10f}")
print(f"signal_other          = {s_other_r1[i_best, j_best]:.10f}")
print(f"signal_full           = {s_full_r1[i_best, j_best]:.10f}")
print(f"background            = {bkg_r1[i_best, j_best]:.10f}")
print()

print("=" * 70)
print(f"Yields in R2 = [{b_best:.6f}, 1]")
print("=" * 70)
print(f"signal_main           = {s_main_r2[i_best, j_best]:.10f}")
print(f"signal_other          = {s_other_r2[i_best, j_best]:.10f}")
print(f"signal_full           = {s_full_r2[i_best, j_best]:.10f}")
print(f"background            = {bkg_r2[i_best, j_best]:.10f}")
print()

# ============================================================
# 10. Direct sanity checks at best thresholds
# ============================================================
print("=" * 70)
print("Direct checks from original dataframes")
print("=" * 70)

# R1 = [a,b)
r1_main_direct = df_sig_main.loc[
    (df_sig_main["bdt_score"] >= a_best) & (df_sig_main["bdt_score"] < b_best),
    "weight"
].sum()

r1_other_direct = df_other.loc[
    (df_other["bdt_score"] >= a_best) & (df_other["bdt_score"] < b_best),
    "weight"
].sum()

r1_bkg_direct = df_bkg.loc[
    (df_bkg["bdt_score"] >= a_best) & (df_bkg["bdt_score"] < b_best),
    "weight"
].sum()

r2_main_direct = df_sig_main.loc[
    df_sig_main["bdt_score"] >= b_best, "weight"
].sum()

r2_other_direct = df_other.loc[
    df_other["bdt_score"] >= b_best, "weight"
].sum()

r2_bkg_direct = df_bkg.loc[
    df_bkg["bdt_score"] >= b_best, "weight"
].sum()

print("R1 direct:")
print(f"signal_main   = {r1_main_direct:.10f}")
print(f"signal_other  = {r1_other_direct:.10f}")
print(f"background    = {r1_bkg_direct:.10f}")
print()

print("R2 direct:")
print(f"signal_main   = {r2_main_direct:.10f}")
print(f"signal_other  = {r2_other_direct:.10f}")
print(f"background    = {r2_bkg_direct:.10f}")
print()

# ============================================================
# 11. Save full scan table if wanted
# ============================================================
ii, jj = np.where(valid)

df_scan = pd.DataFrame({
    "a": thr[ii],
    "b": thr[jj],
    "Z1_opt_main": Z1_opt[ii, jj],
    "Z2_opt_main": Z2_opt[ii, jj],
    "Ztot_opt_main": Ztot_opt[ii, jj],
    "Z1_full": Z1_full[ii, jj],
    "Z2_full": Z2_full[ii, jj],
    "Ztot_full": Ztot_full[ii, jj],
    "sig_main_R1": s_main_r1[ii, jj],
    "sig_other_R1": s_other_r1[ii, jj],
    "sig_full_R1": s_full_r1[ii, jj],
    "bkg_R1": bkg_r1[ii, jj],
    "sig_main_R2": s_main_r2[ii, jj],
    "sig_other_R2": s_other_r2[ii, jj],
    "sig_full_R2": s_full_r2[ii, jj],
    "bkg_R2": bkg_r2[ii, jj],
})

df_scan = df_scan.sort_values("Ztot_opt_main", ascending=False).reset_index(drop=True)
df_scan.to_csv("2Dscan.csv")


print("=" * 70)
print("Top 10 threshold pairs ranked by optimization score (main only)")
print("=" * 70)
print(df_scan.head(10))

# ============================================================
# 12. Plot optimization surface: main-only objective
# ============================================================
plt.figure(figsize=(8, 6))
im = plt.imshow(
    Ztot_opt_masked,
    origin="lower",
    aspect="auto",
    extent=[thr[0], thr[-1], thr[0], thr[-1]]
)
plt.colorbar(im, label="Ztot optimized on signal_main only")
plt.scatter([b_best], [a_best], marker="x", s=100)
plt.xlabel("b threshold")
plt.ylabel("a threshold")
plt.title("Threshold scan optimized on signal_main")
plt.tight_layout()
plt.show()

# ============================================================
# 13. Optional: plot full-signal score evaluated everywhere
# ============================================================
Ztot_full_masked = np.where(valid, Ztot_full, np.nan)

plt.figure(figsize=(8, 6))
im = plt.imshow(
    Ztot_full_masked,
    origin="lower",
    aspect="auto",
    extent=[thr[0], thr[-1], thr[0], thr[-1]]
)
plt.colorbar(im, label="Ztot using signal_main + signal_other")
plt.scatter([b_best], [a_best], marker="x", s=100)
plt.xlabel(f"max(BDT) / {1/400}")
plt.ylabel(f"min(BDT) / {1/400}")
plt.title("Full-signal Z surface, with main-optimized best point marked")
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Prepare inputs
# ============================================================
df_sig_main = df_sig_main.copy()
df_other = df_other.copy()
df_bkg = df_bkg.copy()

# Explicit labels so nothing gets misclassified
df_sig_main["group"] = "signal_main"
df_other["group"] = "signal_other"
df_bkg["group"] = "background"

df_all = pd.concat([df_sig_main, df_other, df_bkg], ignore_index=True)

scores = df_all["bdt_score"].to_numpy()
weights = df_all["weight"].to_numpy()
groups = df_all["group"].to_numpy()

mask_sig_main = (groups == "signal_main")
mask_sig_other = (groups == "signal_other")
mask_bkg = (groups == "background")

# ============================================================
# 2. Asimov significance
# ============================================================
def asimov_array(s, b):
    """
    Vectorized Asimov significance:
        Z = sqrt(2 * [ (s+b) ln(1+s/b) - s ])
    Returns 0 where s<=0 or b<=0.
    """
    out = np.zeros_like(s, dtype=float)
    mask = (s > 0) & (b > 0)
    out[mask] = np.sqrt(
        2.0 * ((s[mask] + b[mask]) * np.log1p(s[mask] / b[mask]) - s[mask])
    )
    return out

def asimov_scalar(s, b):
    if s <= 0 or b <= 0:
        return 0.0
    return np.sqrt(2.0 * ((s + b) * np.log1p(s / b) - s))

# Optional simple significance
def s_over_sqrt_b(s, b):
    if b <= 0:
        return 0.0
    return s / np.sqrt(b)

# ============================================================
# 3. Threshold grid
# ============================================================
# Scan thresholds t in [0,1]
n_thresholds = 401
bins = np.linspace(0.0, 1.0, n_thresholds)
thr = bins[:-1]

# ============================================================
# 4. Histogram weighted yields in score bins
# ============================================================
hist_sig_main, _ = np.histogram(
    scores, bins=bins, weights=weights * mask_sig_main
)
hist_sig_other, _ = np.histogram(
    scores, bins=bins, weights=weights * mask_sig_other
)
hist_bkg, _ = np.histogram(
    scores, bins=bins, weights=weights * mask_bkg
)

# ============================================================
# 5. Cumulative yields above threshold t
# ============================================================
# yield_ge[i] = total weighted yield with score >= thr[i]
sig_main_ge = np.cumsum(hist_sig_main[::-1])[::-1]
sig_other_ge = np.cumsum(hist_sig_other[::-1])[::-1]
bkg_ge = np.cumsum(hist_bkg[::-1])[::-1]

sig_full_ge = sig_main_ge + sig_other_ge

# ============================================================
# 6. Compute significance curves
# ============================================================
# Optimization uses main signal only
Z_opt_main = asimov_array(sig_main_ge, bkg_ge)

# Reported significance uses full signal
Z_full = asimov_array(sig_full_ge, bkg_ge)

# Optional s/sqrt(b) versions
SsqrtB_main = np.array([s_over_sqrt_b(s, b) for s, b in zip(sig_main_ge, bkg_ge)])
SsqrtB_full = np.array([s_over_sqrt_b(s, b) for s, b in zip(sig_full_ge, bkg_ge)])

# ============================================================
# 7. Find best threshold using main-only optimization
# ============================================================
i_best = np.nanargmax(Z_opt_main)
t_best = thr[i_best]

# Yields at best threshold
sig_main_best = sig_main_ge[i_best]
sig_other_best = sig_other_ge[i_best]
sig_full_best = sig_full_ge[i_best]
bkg_best = bkg_ge[i_best]

Z_opt_best = Z_opt_main[i_best]
Z_full_best = Z_full[i_best]

SsqrtB_main_best = SsqrtB_main[i_best]
SsqrtB_full_best = SsqrtB_full[i_best]

# ============================================================
# 8. Print summary
# ============================================================
print("=" * 70)
print("Best threshold from 1D scan")
print("=" * 70)
print(f"t_best                  = {t_best:.6f}")
print()

print("Optimization objective:")
print(f"Z_opt_main              = {Z_opt_best:.6f}")
print()

print("Final reported significance at t_best:")
print(f"Z_full                  = {Z_full_best:.6f}")
print()

print("Optional simple significance at t_best:")
print(f"s/sqrt(b) main only     = {SsqrtB_main_best:.6f}")
print(f"s/sqrt(b) full signal   = {SsqrtB_full_best:.6f}")
print()

print("Yields above threshold:")
print(f"signal_main             = {sig_main_best:.10f}")
print(f"signal_other            = {sig_other_best:.10f}")
print(f"signal_full             = {sig_full_best:.10f}")
print(f"background              = {bkg_best:.10f}")
print()

# ============================================================
# 9. Direct sanity check from original dataframes
# ============================================================
sig_main_direct = df_sig_main.loc[df_sig_main["bdt_score"] >= t_best, "weight"].sum()
sig_other_direct = df_other.loc[df_other["bdt_score"] >= t_best, "weight"].sum()
bkg_direct = df_bkg.loc[df_bkg["bdt_score"] >= t_best, "weight"].sum()

print("=" * 70)
print("Direct sanity check from original dataframes")
print("=" * 70)
print(f"signal_main_direct      = {sig_main_direct:.10f}")
print(f"signal_other_direct     = {sig_other_direct:.10f}")
print(f"background_direct       = {bkg_direct:.10f}")
print()

# ============================================================
# 10. Save scan results in a DataFrame
# ============================================================
df_scan_1d = pd.DataFrame({
    "threshold": thr,
    "sig_main": sig_main_ge,
    "sig_other": sig_other_ge,
    "sig_full": sig_full_ge,
    "background": bkg_ge,
    "Z_opt_main": Z_opt_main,
    "Z_full": Z_full,
    "s_over_sqrt_b_main": SsqrtB_main,
    "s_over_sqrt_b_full": SsqrtB_full,
})

df_scan_1d = df_scan_1d.sort_values("Z_opt_main", ascending=False).reset_index(drop=True)

print("=" * 70)
print("Top 10 thresholds ranked by optimization score (main only)")
print("=" * 70)
print(df_scan_1d.head(10))
print()

# ============================================================
# 11. Plot significance vs threshold
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(thr, Z_opt_main, label="Asimov Z, optimize on signal_main")
plt.plot(thr, Z_full, label="Asimov Z, full signal")
plt.axvline(t_best, linestyle="--", label=f"best t = {t_best:.4f}")

plt.xlabel("BDT threshold t")
plt.ylabel("Significance")
plt.title("1D significance scan")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 12. Plot yields vs threshold
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(thr, sig_main_ge, label="signal_main yield")
plt.plot(thr, sig_other_ge, label="signal_other yield")
plt.plot(thr, sig_full_ge, label="signal_full yield")
plt.plot(thr, bkg_ge, label="background yield")
plt.axvline(t_best, linestyle="--", label=f"best t = {t_best:.4f}")

plt.xlabel("BDT threshold t")
plt.ylabel("Yield above threshold")
plt.title("Yields above threshold vs BDT cut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 13. Optional: plot s/sqrt(b) vs threshold
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(thr, SsqrtB_main, label="s/sqrt(b), main only")
plt.plot(thr, SsqrtB_full, label="s/sqrt(b), full signal")
plt.axvline(t_best, linestyle="--", label=f"best t = {t_best:.4f}")

plt.xlabel("BDT threshold t")
plt.ylabel("s/sqrt(b)")
plt.title("Simple significance vs BDT cut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()