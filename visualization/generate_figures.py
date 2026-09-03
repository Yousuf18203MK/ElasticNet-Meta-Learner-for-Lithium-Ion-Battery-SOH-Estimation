"""
generate_figures.py — Generate all 14 publication-quality figures.

Run from the project root:   python visualization/generate_figures.py
Requires run_all_experiments.py to have completed successfully.
All figures are saved to the figures/ directory at 300 DPI.
"""
import sys
import os
import warnings
import pickle
from pathlib import Path

_SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import RESULTS_DIR, FIGURES_DIR, BATTERIES, LOGS_DIR, DATA_DIR
from src.feature_engineering import extract_rich_features

# ── Journal-quality style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10.5,
    "axes.titlesize":    11.5,
    "axes.labelsize":    10.5,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "legend.fontsize":   9.0,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.7,
})

CELLS   = list(BATTERIES.keys())
PALETTE = {"B0005": "#2166ac", "B0006": "#d6604d", "B0007": "#1a9850", "B0018": "#7b2d8b"}
MODEL_COLORS = {
    "GRU":         "#4d9de0",
    "LSTM":        "#3bb273",
    "XGB":         "#d01c8b",
    "Weighted":    "#8073ac",
    "Stack_LR":    "#e08214",
    "Stack_Ridge": "#35978f",
    "Stack_EN":    "#bf5b17",
}
MODEL_NICE = {
    "GRU":         "GRU",
    "LSTM":        "LSTM",
    "XGB":         "XGBoost",
    "Weighted":    "Weighted\nEns.",
    "Stack_LR":    "Stack\n(LR)",
    "Stack_Ridge": "Stack\n(Ridge)",
    "Stack_EN":    "Stack\n(EN) ★",
}
MODS = list(MODEL_NICE.keys())
NICE = list(MODEL_NICE.values())
MCLR = [MODEL_COLORS[m] for m in MODS]


def load_data():
    """Load all result files needed for figures."""
    full  = pd.read_csv(str(RESULTS_DIR / "final_results_raw.csv"))
    ovrl  = pd.read_csv(str(RESULTS_DIR / "final_overall.csv"))
    smry  = pd.read_csv(str(RESULTS_DIR / "final_summary_per_battery.csv"))
    preds = {}
    for bid in CELLS:
        p = RESULTS_DIR / f"{bid}_predictions_s42.csv"
        if p.exists():
            preds[bid] = pd.read_csv(str(p))
    rich = {}
    for bid, mf in BATTERIES.items():
        rp = RESULTS_DIR / f"{bid}_rich.csv"
        if rp.exists():
            rich[bid] = pd.read_csv(str(rp))
        else:
            print(f"  Extracting features for {bid} …")
            df = extract_rich_features(DATA_DIR / mf, bid)
            df.to_csv(str(rp), index=False)
            rich[bid] = df
    return full, ovrl, smry, preds, rich


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(str(path), dpi=300)
    plt.close(fig)
    print(f"  Saved {name}")


def fig01_degradation(rich):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for ax, bid in zip(axes, CELLS):
        df = rich[bid]; tot = len(df)
        b, e, m = int(tot * .60), int(tot * .70), int(tot * .85)
        ax.fill_betweenx([50, 105], [1, 1], [b, b],   alpha=.12, color="#2166ac", label="Base train")
        ax.fill_betweenx([50, 105], [b, b], [e, e],   alpha=.14, color="#fdae61", label="Early-val")
        ax.fill_betweenx([50, 105], [e, e], [m, m],   alpha=.14, color="#a6dba0", label="Meta block")
        ax.fill_betweenx([50, 105], [m, m], [tot, tot], alpha=.18, color="#762a83", label="Test set")
        ax.plot(df.Cycle, df.SOH, color=PALETTE[bid], lw=2.2, zorder=5)
        ax.set_title(f"{bid}  ({tot} discharge cycles)", fontweight="bold")
        ax.set_xlabel("Discharge cycle"); ax.set_ylabel("SOH (%)")
        ax.set_ylim(50, 103)
    axes[0].legend(loc="lower left", fontsize=7.5, framealpha=.85, ncol=2)
    fig.suptitle("Fig. 1 — SOH Degradation Trajectories with Chronological Data Partition",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig01_degradation_trajectories.png")


def fig02_feature_profiles(rich):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    pairs = [("V_mean","Mean Voltage (V)"), ("V_slope","Voltage Slope (V/sample)"),
             ("T_mean","Mean Temperature (°C)"), ("T_rise","Temperature Rise (°C)"),
             ("power","Mean Power (W)"), ("dur","Discharge Duration (samples)")]
    for ax, (fc, lbl) in zip(axes.flatten(), pairs):
        for bid in CELLS:
            ax.plot(rich[bid].Cycle, rich[bid][fc], color=PALETTE[bid], lw=1.4, alpha=.85, label=bid)
        ax.set_xlabel("Discharge cycle"); ax.set_ylabel(lbl)
    axes[0, 0].legend(fontsize=8, framealpha=.8)
    fig.suptitle("Fig. 2 — Engineered Feature Time Series Across All Four Cells\n"
                 "(within-cycle statistics; Capacity excluded from predictive features)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig02_feature_profiles.png")


def fig03_b0006_diagnostic(rich):
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=.45, wspace=.38)
    df6 = rich["B0006"]; tot6 = len(df6)
    m6, b6 = int(tot6 * .85), int(tot6 * .60)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(df6.Cycle, df6.SOH, "#d6604d", lw=2.2)
    ax0.axvspan(m6, tot6, alpha=.18, color="#762a83", label=f"Test (cycles {m6+1}–{tot6})")
    ax0.axvline(b6, color="#2166ac", lw=1.5, ls="--", label="Base/EV boundary (60%)")
    ax0.axvline(m6, color="#762a83", lw=1.5, ls="--", label="Meta/Test boundary (85%)")
    ax0.set_xlabel("Discharge cycle"); ax0.set_ylabel("SOH (%)")
    ax0.set_title("B0006 SOH Trajectory — Test region spans steepest late-life decline "
                  "(SOH 56.7–63.4%)", fontweight="bold")
    ax0.legend(fontsize=8, ncol=2, framealpha=.8)

    feature_pairs = [
        ("V_mean","V_mean (V)"), ("V_slope","V_slope"), ("T_rise","T_rise (°C)"),
        ("power","Power (W)"),   ("V_p10","V_p10 (V)"), ("dur","Duration"),
    ]
    for ax, (fc, lbl) in zip([fig.add_subplot(gs[i, j]) for i in [1,2] for j in range(3)],
                              feature_pairs):
        ax.plot(df6.Cycle, df6[fc], "#d6604d", lw=1.6, label="B0006")
        for bid2 in ["B0005", "B0007"]:
            ax.plot(rich[bid2].Cycle, rich[bid2][fc], color=PALETTE[bid2], lw=.9, alpha=.55, label=bid2)
        ax.axvspan(m6, tot6, alpha=.12, color="#762a83")
        ax.set_xlabel("Cycle"); ax.set_ylabel(lbl, fontsize=9)
        if fc == "V_mean": ax.legend(fontsize=7.5, framealpha=.7)

    fig.suptitle("Fig. 3 — B0006 Diagnostic Analysis\n"
                 "Purple region = test set; B0006 shows the steepest V decline "
                 "and highest T_rise — largest distribution shift at test time.",
                 fontsize=11, fontweight="bold")
    save(fig, "fig03_b0006_diagnostic.png")


def fig04_actual_vs_predicted(preds):
    if not preds: return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8)); axes = axes.flatten()
    for ax, bid in zip(axes, CELLS):
        if bid not in preds: continue
        p   = preds[bid]
        cyc = p.Cycle.values; act = p.Actual_SOH.values
        ax.plot(cyc, act, "k-", lw=2.4, label="Actual SOH", zorder=5)
        ax.plot(cyc, p.GRU.values,  "--", color="#2166ac", lw=1.3, alpha=.75, label="GRU")
        ax.plot(cyc, p.LSTM.values, "--", color="#4dac26", lw=1.3, alpha=.75, label="LSTM")
        ax.plot(cyc, p.XGB.values,  ":",  color="#d01c8b", lw=1.2, alpha=.55, label="XGBoost")
        ax.plot(cyc, p.Stack_EN.values, "-", color="#bf5b17", lw=2.2,
                label="GRU+LSTM+XGB EN Stack")
        en_rmse = float(np.sqrt(np.mean((act - p.Stack_EN.values)**2)))
        en_mape = float(np.mean(np.abs((act - p.Stack_EN.values) / np.maximum(np.abs(act), 1e-8))) * 100)
        ax.set_title(f"{bid}  |  EN Stack: RMSE={en_rmse:.4f}, MAPE={en_mape:.3f}%",
                     fontweight="bold")
        ax.set_xlabel("Discharge cycle"); ax.set_ylabel("SOH (%)")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=.85, ncol=2)
    fig.suptitle("Fig. 4 — Actual vs Predicted SOH on Test Set (seed=42, representative run)\n"
                 "Proposed: GRU + LSTM + XGBoost → ElasticNet Meta-Learner",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig04_actual_vs_predicted.png")


def fig05_scatter(preds):
    if not preds: return
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5)); axes = axes.flatten()
    for ax, bid in zip(axes, CELLS):
        if bid not in preds: continue
        p   = preds[bid]; act = p.Actual_SOH.values; pred = p.Stack_EN.values
        ax.scatter(act, pred, color=PALETTE[bid], s=48, alpha=.82, edgecolors="white", lw=.6, zorder=5)
        mn = min(act.min(), pred.min()) - .5; mx = max(act.max(), pred.max()) + .5
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.5)
        ss_res = np.sum((act - pred)**2); ss_tot = np.sum((act - np.mean(act))**2)
        r2v    = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.text(.05, .93, f"R² = {r2v:.4f}", transform=ax.transAxes, fontsize=10,
                bbox=dict(fc="white", alpha=.88, boxstyle="round,pad=0.3"))
        ax.set_title(f"{bid}", fontweight="bold")
        ax.set_xlabel("Actual SOH (%)"); ax.set_ylabel("Predicted SOH (%)")
    fig.suptitle("Fig. 5 — Predicted vs Actual SOH (seed=42)\nGRU+LSTM+XGBoost ElasticNet Stack",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig05_scatter.png")


def fig06_residuals(preds):
    if not preds: return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5)); axes = axes.flatten()
    for ax, bid in zip(axes, CELLS):
        if bid not in preds: continue
        p   = preds[bid]; res = p.Stack_EN.values - p.Actual_SOH.values
        ax.bar(p.Cycle.values, res, color=PALETTE[bid], alpha=.78, width=.65)
        ax.axhline(y=0,              color="k",    lw=1.3, ls="--")
        ax.axhline(y=float(res.std()),  color="grey", lw=1,   ls=":", label=f"σ={res.std():.4f}")
        ax.axhline(y=-float(res.std()), color="grey", lw=1,   ls=":")
        ax.set_title(f"{bid}  |  μ={res.mean():.4f}  σ={res.std():.4f}", fontweight="bold")
        ax.set_xlabel("Discharge cycle"); ax.set_ylabel("Residual (SOH %)")
        ax.legend(fontsize=8.5, framealpha=.8)
    fig.suptitle("Fig. 6 — Prediction Residuals (seed=42)\nGRU+LSTM+XGBoost ElasticNet Stack",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig06_residuals.png")


def fig07_per_cell_rmse(full, smry):
    x = np.arange(len(MODS)); w = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    for i, (bid, clr) in enumerate(PALETTE.items()):
        vals = [smry[(smry.Battery==bid) & (smry.Model==m)].RMSE_mean.values
                for m in MODS]
        errs = [smry[(smry.Battery==bid) & (smry.Model==m)].RMSE_std.values
                for m in MODS]
        v = [float(v[0]) if len(v) else 0.0 for v in vals]
        e = [float(v[0]) if len(v) else 0.0 for v in errs]
        ax.bar(x + i*w, v, w, label=bid, color=clr, alpha=.85,
               yerr=e, capsize=3, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(NICE, fontsize=9.5)
    ax.set_ylabel("RMSE (SOH %)")
    ax.set_title("Fig. 7 — Per-Cell RMSE: All Models (mean ± std, 3 seeds)\n★ = Proposed method",
                 fontweight="bold")
    ax.legend(ncol=4, fontsize=9, loc="upper right", framealpha=.85)
    ax.axvspan(5.7, 7.3, alpha=.07, color="gold", zorder=0)
    plt.tight_layout()
    save(fig, "fig07_rmse_per_cell.png")


def fig08_overall_comparison(ovrl):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    for ax, (met, ylab) in zip(axes,
            [("RMSE_mean","RMSE (SOH %)"),
             ("MAE_mean", "MAE (SOH %)"),
             ("MAPE_mean","MAPE (%)")]):
        vals = []
        for mdl in MODS:
            row = ovrl[ovrl.Model == mdl]
            vals.append(float(row[met].values[0]) if len(row) else 0.0)
        bars = ax.bar(NICE, vals, color=MCLR, alpha=.87)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + .05,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_title(met.replace("_mean", ""), fontweight="bold")
        mi = int(np.argmin(vals))
        bars[mi].set_edgecolor("black"); bars[mi].set_linewidth(2.2)
    fig.suptitle("Fig. 8 — Overall Model Performance (mean across 4 cells × 3 seeds)\n"
                 "Black border = best model", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig08_overall_comparison.png")


def fig09_rmse_heatmap(smry):
    mat = np.zeros((len(MODS), len(CELLS)))
    for i, mdl in enumerate(MODS):
        for j, bid in enumerate(CELLS):
            row = smry[(smry.Battery==bid) & (smry.Model==mdl)]
            mat[i, j] = float(row.RMSE_mean.values[0]) if len(row) else np.nan
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(CELLS, fontsize=11)
    ax.set_yticks(range(len(MODS))); ax.set_yticklabels(NICE, fontsize=10)
    for i in range(len(MODS)):
        for j in range(4):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 5 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="RMSE (SOH %)", fraction=.025)
    ax.set_title("Fig. 9 — RMSE Heatmap (mean, 3 seeds)", fontweight="bold")
    plt.tight_layout()
    save(fig, "fig09_rmse_heatmap.png")


def fig10_multiseed(full):
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    for ax, bid in zip(axes, CELLS):
        sub  = full[full.Battery == bid]
        sv   = sub["Stack_EN_RMSE"].values
        seeds = sub["Seed"].astype(str).values
        ax.bar(seeds, sv, color=PALETTE[bid], alpha=.85, edgecolor="white")
        ax.axhline(y=float(sv.mean()), color="k", lw=1.5, ls="--",
                   label=f"μ={sv.mean():.3f}")
        ax.set_title(f"{bid}", fontweight="bold")
        ax.set_xlabel("Seed")
        ax.set_ylabel("RMSE (SOH %)" if bid == CELLS[0] else "")
        ax.legend(fontsize=8.5, framealpha=.8)
    fig.suptitle("Fig. 10 — Seed-Wise RMSE: GRU+LSTM+XGBoost EN Stack",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig10_multiseed.png")


def fig11_b0018_diagnostic(full, preds, smry):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    b18  = smry[smry.Battery == "B0018"]
    vals = [float(b18[b18.Model==m].RMSE_mean.values[0])
            if len(b18[b18.Model==m]) else 0.0 for m in MODS]
    errs = [float(b18[b18.Model==m].RMSE_std.values[0])
            if len(b18[b18.Model==m]) else 0.0 for m in MODS]
    bars = axes[0].bar(NICE, vals, color=MCLR, alpha=.87, yerr=errs, capsize=4)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + .04,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylabel("RMSE (SOH %)")
    axes[0].set_title("B0018 RMSE by Model\n(mean ± std, 3 seeds)", fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=8.5)

    if "B0018" in preds:
        p18 = preds["B0018"]
        axes[1].plot(p18.Cycle, p18.Actual_SOH,   "k-",  lw=2.2, label="Actual")
        axes[1].plot(p18.Cycle, p18.Stack_EN, "#bf5b17", lw=2.0,
                     label=f"EN Stack (RMSE="
                           f"{float(np.sqrt(np.mean((p18.Actual_SOH - p18.Stack_EN)**2))):.4f})")
        axes[1].plot(p18.Cycle, p18.GRU,  "--", color="#2166ac", lw=1.3, alpha=.75, label="GRU")
        axes[1].plot(p18.Cycle, p18.LSTM, "--", color="#4dac26", lw=1.3, alpha=.75, label="LSTM")
        axes[1].set_xlabel("Discharge cycle"); axes[1].set_ylabel("SOH (%)")
        axes[1].set_title("B0018 Actual vs Predicted (seed=42)", fontweight="bold")
        axes[1].legend(fontsize=8.5)

    fig.suptitle("Fig. 11 — B0018 Diagnostic: Short Degradation Record (132 cycles, 20 test seqs)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig11_b0018_diagnostic.png")


def fig12_residual_histograms(preds):
    if not preds: return
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    for ax, bid in zip(axes, CELLS):
        if bid not in preds: continue
        p   = preds[bid]; res = p.Stack_EN.values - p.Actual_SOH.values
        ax.hist(res, bins=8, color=PALETTE[bid], alpha=.82, edgecolor="white")
        ax.axvline(x=0,              color="k",   lw=1.5, ls="--")
        ax.axvline(x=float(res.mean()), color="red", lw=1.5,
                   label=f"μ={res.mean():.3f}")
        ax.set_title(f"{bid}", fontweight="bold")
        ax.set_xlabel("Residual (SOH %)")
        if bid == CELLS[0]: ax.set_ylabel("Count")
        ax.legend(fontsize=8.5)
    fig.suptitle("Fig. 12 — Residual Distributions (GRU+LSTM+XGB EN Stack, seed=42)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig12_residual_hist.png")


def fig13_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8)); axes = axes.flatten()
    for ax, bid in zip(axes, CELLS):
        for mdl, clr, ls in [("LSTM","#3bb273","-"), ("GRU","#4d9de0","--")]:
            log = LOGS_DIR / f"{bid}_s42_{mdl}.csv"
            if log.exists():
                h = pd.read_csv(str(log))
                ep = range(1, len(h)+1)
                ax.plot(ep, h.loss,     color=clr, lw=1.8, ls=ls,    label=f"{mdl} train")
                ax.plot(ep, h.val_loss, color=clr, lw=1.4, ls=":",   label=f"{mdl} val", alpha=.75)
        ax.set_title(f"{bid}", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8, framealpha=.8)
    fig.suptitle("Fig. 13 — Training and Validation Loss Curves (seed=42)\n"
                 "EarlyStopping monitors early-val partition only (test set never touched)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig13_learning_curves.png")


def fig14_ablation(full):
    # Build ablation mean values from final_results_raw.csv
    configs = ["GRU", "LSTM", "XGB", "Weighted", "Stack_LR", "Stack_Ridge", "Stack_EN"]
    labels  = ["GRU", "LSTM", "XGBoost", "Weighted\nEns.", "Stack\n(LR)", "Stack\n(Ridge)", "Stack\n(EN)★"]
    colors  = [MODEL_COLORS.get(c, "#888888") for c in configs]

    vals_per_bid = {bid: [] for bid in CELLS}
    for bid in CELLS:
        sub = full[full.Battery == bid]
        for cfg in configs:
            col = f"{cfg}_RMSE"
            vals_per_bid[bid].append(float(sub[col].mean()) if col in sub.columns else 0.0)

    x = np.arange(len(configs)); w = 0.19
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (bid, clr) in enumerate(PALETTE.items()):
        ax.bar(x + i*w, vals_per_bid[bid], w, label=bid, color=clr, alpha=.85)
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("RMSE (SOH %)")
    ax.set_title("Fig. 14 — Ablation Study: Per-Cell RMSE Across All Configurations\n"
                 "(mean, 3 seeds; ★ = Proposed final method)", fontweight="bold")
    ax.legend(ncol=4, fontsize=9, loc="upper right", framealpha=.85)
    ax.axvspan(5.7, 7.3, alpha=.06, color="gold", zorder=0)
    plt.tight_layout()
    save(fig, "fig14_ablation.png")


def main():
    print("=" * 60)
    print("  Generating publication figures …")
    print("=" * 60)

    # Check results exist
    if not (RESULTS_DIR / "final_results_raw.csv").exists():
        raise FileNotFoundError(
            "final_results_raw.csv not found.\n"
            "Run experiments/run_all_experiments.py first."
        )

    full, ovrl, smry, preds, rich = load_data()

    fig01_degradation(rich)
    fig02_feature_profiles(rich)
    fig03_b0006_diagnostic(rich)
    fig04_actual_vs_predicted(preds)
    fig05_scatter(preds)
    fig06_residuals(preds)
    fig07_per_cell_rmse(full, smry)
    fig08_overall_comparison(ovrl)
    fig09_rmse_heatmap(smry)
    fig10_multiseed(full)
    fig11_b0018_diagnostic(full, preds, smry)
    fig12_residual_histograms(preds)
    fig13_learning_curves()
    fig14_ablation(full)

    print(f"\n  All 14 figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
