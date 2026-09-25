
import sys
import os
import warnings
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import RESULTS_DIR, FIGURES_DIR, BATTERIES, LOGS_DIR, DATA_DIR
from src.feature_engineering import extract_rich_features

CELLS = list(BATTERIES.keys())
SEEDS = [42, 123, 2024]

PALETTE = {
    "B0005": "#2166ac",
    "B0006": "#d6604d",
    "B0007": "#1a9850",
    "B0018": "#7b2d8b",
}

MODEL_COLORS = {
    "GRU": "#4d9de0",
    "LSTM": "#3bb273",
    "XGB": "#d01c8b",
    "Weighted": "#8073ac",
    "Stack_LR": "#e08214",
    "Stack_Ridge": "#35978f",
    "Stack_EN": "#bf5b17",
}

MODS = ["GRU", "LSTM", "XGB", "Weighted", "Stack_LR", "Stack_Ridge", "Stack_EN"]
MODS_LOO = ["GRU", "LSTM", "XGB", "Weighted", "Stack_Ridge", "Stack_EN"]
NICE = ["GRU", "LSTM", "XGBoost", "Weighted\nEns.", "Stack\nLR", "Stack\nRidge", "Stack\nEN"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "lines.linewidth": 1.7,
})

LOO_DIR = RESULTS_DIR / "loo"


def save(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {filename}")


def load_all():
    required = [
        RESULTS_DIR / "final_results_raw.csv",
        RESULTS_DIR / "final_overall.csv",
        RESULTS_DIR / "final_summary_per_battery.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required result file: {p}")

    full = pd.read_csv(RESULTS_DIR / "final_results_raw.csv")
    overall = pd.read_csv(RESULTS_DIR / "final_overall.csv")
    summary = pd.read_csv(RESULTS_DIR / "final_summary_per_battery.csv")

    preds = {}
    rich = {}

    for bid in CELLS:
        pred_file = RESULTS_DIR / f"{bid}_predictions_s42.csv"
        if pred_file.exists():
            preds[bid] = pd.read_csv(pred_file)

        rich_file = RESULTS_DIR / f"{bid}_rich.csv"
        if rich_file.exists():
            rich[bid] = pd.read_csv(rich_file)
        else:
            mat_file = DATA_DIR / BATTERIES[bid]
            rich[bid] = extract_rich_features(mat_file, bid)
            rich[bid].to_csv(rich_file, index=False)

    return full, overall, summary, preds, rich


def load_loo():
    required = [
        LOO_DIR / "loo_results_raw.csv",
        LOO_DIR / "loo_summary_per_battery.csv",
        LOO_DIR / "loo_overall.csv",
        LOO_DIR / "loo_preds_s42.pkl",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing LOO result file: {p}\n"
                "Run experiments/run_loo_experiment.py first."
            )

    loo_raw = pd.read_csv(LOO_DIR / "loo_results_raw.csv")
    loo_summary = pd.read_csv(LOO_DIR / "loo_summary_per_battery.csv")
    loo_overall = pd.read_csv(LOO_DIR / "loo_overall.csv")

    with open(LOO_DIR / "loo_preds_s42.pkl", "rb") as f:
        loo_preds = pickle.load(f)

    return loo_raw, loo_summary, loo_overall, loo_preds


def add_panel_label(ax, label):
    ax.text(
        0.015, 0.965, f"({label})",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )


def make_box(ax, x, y, w, h, title, body, face):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="white",
        facecolor=face,
        zorder=3
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h * 0.64, title,
        ha="center", va="center",
        color="white", fontsize=9, fontweight="bold", zorder=4
    )
    ax.text(
        x + w / 2, y + h * 0.28, body,
        ha="center", va="center",
        color="white", fontsize=8, linespacing=1.25, zorder=4
    )


def add_arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        color="#4B5563",
        connectionstyle=f"arc3,rad={rad}",
        zorder=2
    ))


def figure_1(rich):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, bid in zip(axes, CELLS):
        df = rich[bid]
        total = len(df)

        base_end = int(total * 0.60)
        ev_end = int(total * 0.70)
        meta_end = int(total * 0.85)

        ax.axvspan(0, base_end, color="#2166ac", alpha=0.10, label="Base training, 60%")
        ax.axvspan(base_end, ev_end, color="#fdae61", alpha=0.16, label="Early validation, 10%")
        ax.axvspan(ev_end, meta_end, color="#35a979", alpha=0.14, label="Meta training, 15%")
        ax.axvspan(meta_end, total, color="#762a83", alpha=0.14, label="Final test set, 15%")

        ax.plot(
            df.Cycle, df.SOH,
            color=PALETTE[bid], lw=2.2,
            label="Measured SOH trajectory", zorder=5
        )

        ax.set_title(
            f"{bid}  ({total} discharge cycles)",
            fontweight="bold"
        )
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("SOH (%)")
        ax.set_ylim(50, 103)

        ax.text(
            0.015, 0.04,
            "Background: chronological partition",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none")
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        fontsize=8.5
    )

    fig.suptitle(
        "Figure 1. SOH degradation trajectories and chronological data partition",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    save(fig, "fig01_degradation_trajectories.png")


def figure_2_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    navy = "#203D70"
    blue = "#2774B9"
    teal = "#319B95"
    grey = "#555A60"
    green = "#1F5C3A"
    orange = "#D98B3A"
    purple = "#752A84"
    light_blue = "#4A9DDA"
    light_green = "#35A979"
    magenta = "#D11879"
    brown = "#B65E16"

    ax.text(
        6.75, 7.72,
        "Figure 2. Experimental workflow and model architecture",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=navy
    )

    make_box(ax, 0.35, 6.25, 1.75, 0.95, "NASA MAT files", "B0005 to B0018", navy)
    make_box(ax, 2.45, 6.25, 1.75, 0.95, "Discharge extraction", "V, I, T, Capacity", blue)
    make_box(ax, 4.55, 6.25, 1.85, 0.95, "20 feature extraction", "per cycle", teal)
    make_box(ax, 6.75, 6.25, 1.65, 0.95, "MinMax scaler", "fit on training only", grey)
    make_box(ax, 8.75, 6.25, 2.05, 0.95, "Sliding window", "sequence length = 32\nXGB lag = 5", navy)

    add_arrow(ax, 2.10, 6.73, 2.45, 6.73)
    add_arrow(ax, 4.20, 6.73, 4.55, 6.73)
    add_arrow(ax, 6.40, 6.73, 6.75, 6.73)
    add_arrow(ax, 8.40, 6.73, 8.75, 6.73)

    ax.text(
        5.60, 6.05,
        "Data ingestion and preprocessing",
        ha="center", fontsize=8.5, color="#555555", style="italic"
    )

    make_box(ax, 0.35, 4.60, 1.95, 0.82, "Chronological split", "60 / 10 / 15 / 15%", green)
    make_box(ax, 2.55, 4.60, 1.65, 0.82, "Base training", "60%", navy)
    make_box(ax, 4.45, 4.60, 1.35, 0.82, "Early validation", "10%", orange)
    make_box(ax, 6.05, 4.60, 1.45, 0.82, "Meta training", "15%", light_green)
    make_box(ax, 7.75, 4.60, 1.35, 0.82, "Final test", "15%", purple)

    add_arrow(ax, 2.30, 5.01, 2.55, 5.01)
    add_arrow(ax, 4.20, 5.01, 4.45, 5.01)
    add_arrow(ax, 5.80, 5.01, 6.05, 5.01)
    add_arrow(ax, 7.50, 5.01, 7.75, 5.01)

    ax.text(
        5.05, 4.27,
        "Strictly chronological, no future data, leakage controlled",
        ha="center", fontsize=8.3, color="#555555", style="italic"
    )

    make_box(ax, 3.35, 3.00, 1.85, 0.85, "GRU", "2 layers, 64 units\ndropout = 0.20", light_blue)
    make_box(ax, 5.70, 3.00, 1.85, 0.85, "LSTM", "2 layers, 64 units\ndropout = 0.20", light_green)
    make_box(ax, 8.05, 3.00, 1.85, 0.85, "XGBoost", "300 trees, depth = 4\nlag features", magenta)

    add_arrow(ax, 3.38, 4.60, 3.38, 4.05)
    ax.plot([3.38, 8.98], [4.05, 4.05], color="#4B5563", linewidth=1.2, zorder=1)
    add_arrow(ax, 4.28, 4.05, 4.28, 3.86)
    add_arrow(ax, 6.63, 4.05, 6.63, 3.86)
    add_arrow(ax, 8.98, 4.05, 8.98, 3.86)

    make_box(
        ax, 4.70, 1.70, 3.90, 0.82,
        "Meta predictions to ElasticNet stacking",
        "alpha = 0.05, l1 ratio = 0.50",
        brown
    )

    add_arrow(ax, 4.28, 3.00, 5.25, 2.52, rad=0.02)
    add_arrow(ax, 6.63, 3.00, 6.65, 2.52)
    add_arrow(ax, 8.98, 3.00, 7.95, 2.52, rad=-0.02)

    make_box(
        ax, 9.35, 1.70, 3.45, 0.82,
        "Final test evaluation",
        "held out from model fitting",
        navy
    )

    add_arrow(ax, 8.42, 5.01, 12.10, 4.20, rad=-0.08)
    add_arrow(ax, 12.10, 4.20, 12.10, 2.52)

    ax.text(
        6.75, 0.78,
        "LOO: one complete battery is held out; the remaining batteries supply training and meta data",
        ha="center", fontsize=8.2, color="#555555", style="italic"
    )

    fig.savefig(
        FIGURES_DIR / "fig02_workflow.png",
        dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print("Saved fig02_workflow.png")


def figure_3_actual_vs_predicted(preds):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, bid in zip(axes, CELLS):
        if bid not in preds:
            continue

        p = preds[bid]
        actual = p.Actual_SOH.values
        pred = p.Stack_EN.values

        ax.plot(p.Cycle, actual, "k-", lw=2.4, label="Actual SOH")
        ax.plot(p.Cycle, p.GRU, "--", color="#2166ac", lw=1.3, label="GRU")
        ax.plot(p.Cycle, p.LSTM, "--", color="#4dac26", lw=1.3, label="LSTM")
        ax.plot(p.Cycle, p.XGB, ":", color="#d01c8b", lw=1.3, label="XGBoost")
        ax.plot(
            p.Cycle, pred,
            "-", color="#bf5b17", lw=2.2,
            label="GRU + LSTM + XGBoost + ElasticNet"
        )

        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(
            np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))
        ) * 100

        ax.set_title(
            f"{bid} | RMSE = {rmse:.3f}, MAPE = {mape:.2f}%",
            fontweight="bold"
        )
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("SOH (%)")

    axes[0].legend(
        fontsize=7.7,
        loc="lower left",
        framealpha=0.88,
        ncol=2
    )

    fig.suptitle(
        "Figure 3. Actual versus predicted SOH on within battery test sets",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig03_actual_vs_predicted.png")


def figure_4_heatmap(summary):
    mat = np.full((len(MODS), len(CELLS)), np.nan)

    for i, model in enumerate(MODS):
        for j, bid in enumerate(CELLS):
            row = summary[
                (summary.Battery == bid) &
                (summary.Model == model)
            ]
            if len(row):
                mat[i, j] = float(row.RMSE_mean.iloc[0])

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=8)

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELLS)
    ax.set_yticks(range(len(MODS)))
    ax.set_yticklabels(NICE)

    for i in range(len(MODS)):
        for j in range(len(CELLS)):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(
                    j, i, f"{value:.3f}",
                    ha="center", va="center",
                    fontsize=9.2, fontweight="bold",
                    color="white" if value > 5 else "black"
                )

    ax.set_xlabel("Battery cell")
    ax.set_ylabel("Model or ensemble configuration")
    ax.set_title(
        "Figure 4. Within battery RMSE heatmap, mean across three seeds",
        fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="RMSE, SOH percentage points")
    plt.tight_layout()
    save(fig, "fig04_rmse_heatmap.png")


def figure_5_multiseed(full):
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))

    for ax, bid in zip(axes, CELLS):
        sub = full[full.Battery == bid].copy()
        values = sub["Stack_EN_RMSE"].values
        seeds = sub["Seed"].astype(str).values

        ax.bar(
            seeds, values,
            color=PALETTE[bid],
            alpha=0.85,
            edgecolor="white"
        )

        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1))

        ax.axhline(
            mean_value,
            color="black",
            lw=1.5,
            ls="--",
            label=f"Three seed mean = {mean_value:.3f}"
        )

        ax.text(
            0.03, 0.92,
            f"Bars: Stack ElasticNet RMSE\n"
            f"Dashed line: three seed mean\n"
            f"SD = {std_value:.3f}",
            transform=ax.transAxes,
            fontsize=7.8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none")
        )

        ax.set_title(bid, fontweight="bold")
        ax.set_xlabel("Evaluation seed")
        ax.set_ylabel("RMSE, SOH %" if bid == CELLS[0] else "")
        ax.legend(fontsize=7.6, framealpha=0.88)

    fig.suptitle(
        "Figure 5. Seed wise Stack ElasticNet RMSE",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig05_multiseed_rmse.png")


def figure_6_residual_bars(preds):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, bid in zip(axes, CELLS):
        p = preds[bid]
        residual = p.Stack_EN.values - p.Actual_SOH.values
        sigma = float(np.std(residual, ddof=1))
        mu = float(np.mean(residual))

        ax.bar(
            p.Cycle.values,
            residual,
            color=PALETTE[bid],
            alpha=0.78,
            width=0.65
        )
        ax.axhline(0, color="black", lw=1.3, ls="--", label="Zero residual")
        ax.axhline(
            sigma, color="grey", lw=1.0, ls=":",
            label=f"+1 SD = {sigma:.3f}"
        )
        ax.axhline(-sigma, color="grey", lw=1.0, ls=":")

        ax.set_title(
            f"{bid} | mean residual = {mu:.3f}, SD = {sigma:.3f}",
            fontweight="bold"
        )
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("Residual, predicted minus actual SOH (%)")
        ax.legend(fontsize=7.7, framealpha=0.88)

    fig.suptitle(
        "Figure 6. Cycle level prediction residuals for Stack ElasticNet, seed 42",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig06_residuals.png")


def figure_7_residual_hist(preds):
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))

    for ax, bid in zip(axes, CELLS):
        p = preds[bid]
        residual = p.Stack_EN.values - p.Actual_SOH.values
        mean_value = float(np.mean(residual))
        std_value = float(np.std(residual, ddof=1))

        ax.hist(
            residual,
            bins=8,
            color=PALETTE[bid],
            alpha=0.82,
            edgecolor="white"
        )
        ax.axvline(
            0, color="black", lw=1.5, ls="--",
            label="Zero residual"
        )
        ax.axvline(
            mean_value, color="red", lw=1.5,
            label=f"Mean = {mean_value:.3f}"
        )

        ax.set_title(
            f"{bid} | SD = {std_value:.3f}",
            fontweight="bold"
        )
        ax.set_xlabel("Residual, predicted minus actual SOH (%)")
        if bid == CELLS[0]:
            ax.set_ylabel("Count")
        ax.legend(fontsize=7.6, framealpha=0.88)

    fig.suptitle(
        "Figure 7. Residual distributions for Stack ElasticNet, seed 42",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig07_residual_hist.png")


def figure_8_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, bid in zip(axes, CELLS):
        found = False

        for model, color, linestyle in [
            ("GRU", "#4d9de0", "--"),
            ("LSTM", "#3bb273", "-"),
        ]:
            path = LOGS_DIR / f"{bid}_s42_{model}.csv"

            if not path.exists():
                continue

            history = pd.read_csv(path)

            if "loss" not in history.columns or "val_loss" not in history.columns:
                continue

            epochs = np.arange(1, len(history) + 1)

            ax.plot(
                epochs,
                history["loss"],
                color=color,
                lw=1.8,
                ls=linestyle,
                label=f"{model} training loss"
            )
            ax.plot(
                epochs,
                history["val_loss"],
                color=color,
                lw=1.4,
                ls=":",
                alpha=0.78,
                label=f"{model} validation loss"
            )
            found = True

        ax.set_title(bid, fontweight="bold")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("MSE loss")
        if found:
            ax.legend(fontsize=7.7, framealpha=0.88)

    fig.suptitle(
        "Figure 8. Training and validation loss curves, seed 42",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig08_learning_curves.png")


def figure_9_loo_predictions(loo_preds, rich):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, bid in zip(axes, CELLS):
        p = loo_preds[bid]
        actual = p.Actual_SOH.values
        stack = p.Stack_EN.values

        ax.plot(
            rich[bid].Cycle,
            rich[bid].SOH,
            color="lightgrey",
            lw=1.0,
            label="Full SOH trajectory"
        )
        ax.plot(
            p.Cycle,
            actual,
            "k-",
            lw=2.4,
            label="Actual LOO test SOH"
        )
        ax.plot(
            p.Cycle,
            p.GRU,
            "--",
            color="#2166ac",
            lw=1.3,
            label="GRU"
        )
        ax.plot(
            p.Cycle,
            p.LSTM,
            "--",
            color="#4dac26",
            lw=1.3,
            label="LSTM"
        )
        ax.plot(
            p.Cycle,
            stack,
            "-",
            color="#bf5b17",
            lw=2.2,
            label="Stack ElasticNet"
        )

        rmse = np.sqrt(np.mean((actual - stack) ** 2))

        ax.set_title(
            f"{bid} | LOO RMSE = {rmse:.3f}",
            fontweight="bold"
        )
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("SOH (%)")

    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    add_panel_label(axes[2], "c")
    add_panel_label(axes[3], "d")

    axes[0].legend(
        fontsize=7.4,
        loc="lower left",
        framealpha=0.88,
        ncol=2
    )

    fig.suptitle(
        "Figure 9. Leave One Battery Out actual versus predicted SOH, seed 42",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig09_loo_actual_vs_predicted.png")


def figure_10_within_vs_loo(full, loo_overall, loo_summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    wb_values = []
    loo_values = []

    for model in MODS_LOO:
        wb_col = f"{model}_RMSE"

        if wb_col in full.columns:
            wb_values.append(float(full[wb_col].mean()))
        else:
            wb_values.append(np.nan)

        row = loo_overall[loo_overall.Model == model]
        loo_values.append(
            float(row.RMSE_mean.iloc[0]) if len(row) else np.nan
        )

    x = np.arange(len(MODS_LOO))
    width = 0.36

    axes[0].bar(
        x - width / 2,
        wb_values,
        width,
        label="Within battery",
        color=[MODEL_COLORS[m] for m in MODS_LOO],
        alpha=0.86
    )
    axes[0].bar(
        x + width / 2,
        loo_values,
        width,
        label="Leave One Battery Out",
        color=[MODEL_COLORS[m] for m in MODS_LOO],
        alpha=0.42,
        hatch="////",
        edgecolor="grey"
    )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["GRU", "LSTM", "XGBoost", "Weighted", "Stack Ridge", "Stack EN"],
        fontsize=8.8
    )
    axes[0].set_ylabel("Mean RMSE, SOH percentage points")
    axes[0].set_title(
        "Model level comparison",
        fontweight="bold"
    )
    axes[0].legend(fontsize=8.6)

    within_per = full.groupby("Battery")["Stack_EN_RMSE"].mean()

    loo_per = {}
    for bid in CELLS:
        row = loo_summary[
            (loo_summary.Test_Battery == bid) &
            (loo_summary.Model == "Stack_EN")
        ]
        loo_per[bid] = float(row.RMSE_mean.iloc[0]) if len(row) else np.nan

    x2 = np.arange(len(CELLS))

    axes[1].bar(
        x2 - width / 2,
        [within_per[b] for b in CELLS],
        width,
        color=[PALETTE[b] for b in CELLS],
        alpha=0.86,
        label="Within battery"
    )
    axes[1].bar(
        x2 + width / 2,
        [loo_per[b] for b in CELLS],
        width,
        color=[PALETTE[b] for b in CELLS],
        alpha=0.42,
        hatch="////",
        edgecolor="grey",
        label="LOO"
    )

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(CELLS)
    axes[1].set_ylabel("RMSE, SOH percentage points")
    axes[1].set_title(
        "Stack ElasticNet by held out battery",
        fontweight="bold"
    )
    axes[1].legend(fontsize=8.6)

    fig.suptitle(
        "Figure 10. Within battery versus Leave One Battery Out performance",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig10_within_vs_loo.png")


def figure_11_b0006(loo_raw, loo_preds, rich):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.15, 1],
        hspace=0.42,
        wspace=0.34
    )

    p = loo_preds["B0006"]
    df = rich["B0006"]

    ax0 = fig.add_subplot(gs[0, :])

    ax0.plot(
        df.Cycle,
        df.SOH,
        color="lightgrey",
        lw=1.0,
        label="Full SOH trajectory"
    )
    ax0.plot(
        p.Cycle,
        p.Actual_SOH,
        color="#d6604d",
        lw=2.4,
        label="Actual LOO test SOH"
    )
    ax0.plot(
        p.Cycle,
        p.Stack_EN,
        color="#bf5b17",
        lw=2.2,
        label="Stack ElasticNet"
    )
    ax0.plot(
        p.Cycle,
        p.GRU,
        "--",
        color="#2166ac",
        lw=1.3,
        label="GRU"
    )
    ax0.plot(
        p.Cycle,
        p.LSTM,
        "--",
        color="#4dac26",
        lw=1.3,
        label="LSTM"
    )

    ax0.set_xlabel("Discharge cycle")
    ax0.set_ylabel("SOH (%)")
    ax0.set_title(
        "B0006 LOO, seed 42, training cells: B0005, B0007, B0018",
        fontweight="bold"
    )
    ax0.legend(fontsize=8.0, ncol=3, framealpha=0.88)
    add_panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1, 0])
    sub = loo_raw[loo_raw.Test_Battery == "B0006"].copy()

    ax1.bar(
        sub.Seed.astype(str),
        sub.Stack_EN_RMSE,
        color="#d6604d",
        alpha=0.85,
        edgecolor="white"
    )

    mean_rmse = float(sub.Stack_EN_RMSE.mean())
    std_rmse = float(sub.Stack_EN_RMSE.std(ddof=1))

    ax1.axhline(
        mean_rmse,
        color="black",
        lw=1.5,
        ls="--",
        label=f"Three seed mean = {mean_rmse:.3f}"
    )

    ax1.text(
        0.03, 0.92,
        f"Bars: Stack ElasticNet RMSE\n"
        f"SD = {std_rmse:.3f}",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="none")
    )

    ax1.set_xlabel("Evaluation seed")
    ax1.set_ylabel("RMSE, SOH %")
    ax1.set_title(
        "B0006 LOO seed stability",
        fontweight="bold"
    )
    ax1.legend(fontsize=8)
    add_panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, 1])
    residual = p.Stack_EN.values - p.Actual_SOH.values

    ax2.scatter(
        p.Actual_SOH,
        residual,
        color="#d6604d",
        s=35,
        alpha=0.78
    )
    ax2.axhline(
        0,
        color="black",
        lw=1.2,
        ls="--",
        label="Zero residual"
    )

    ax2.set_xlabel("Actual SOH (%)")
    ax2.set_ylabel("Residual, predicted minus actual SOH (%)")
    ax2.set_title(
        "B0006 residual versus actual SOH",
        fontweight="bold"
    )
    ax2.legend(fontsize=8)
    add_panel_label(ax2, "c")

    fig.suptitle(
        "Figure 11. B0006 Leave One Battery Out diagnostic",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig11_b0006_loo_diagnostic.png")


def figure_12_b0018(loo_raw, loo_preds, rich):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    p = loo_preds["B0018"]
    df = rich["B0018"]

    axes[0].plot(
        df.Cycle,
        df.SOH,
        color="lightgrey",
        lw=1.0,
        label="Full SOH trajectory"
    )
    axes[0].plot(
        p.Cycle,
        p.Actual_SOH,
        color="#7b2d8b",
        lw=2.4,
        label="Actual LOO test SOH"
    )
    axes[0].plot(
        p.Cycle,
        p.Stack_EN,
        color="#bf5b17",
        lw=2.2,
        label="Stack ElasticNet"
    )
    axes[0].plot(
        p.Cycle,
        p.GRU,
        "--",
        color="#2166ac",
        lw=1.3,
        label="GRU"
    )
    axes[0].plot(
        p.Cycle,
        p.LSTM,
        "--",
        color="#4dac26",
        lw=1.3,
        label="LSTM"
    )

    axes[0].set_xlabel("Discharge cycle")
    axes[0].set_ylabel("SOH (%)")
    axes[0].set_title(
        "B0018 LOO, seed 42, training cells: B0005, B0006, B0007",
        fontweight="bold"
    )
    axes[0].legend(fontsize=7.8, framealpha=0.88)
    add_panel_label(axes[0], "a")

    sub = loo_raw[loo_raw.Test_Battery == "B0018"].copy()

    axes[1].bar(
        sub.Seed.astype(str),
        sub.Stack_EN_RMSE,
        color="#7b2d8b",
        alpha=0.85,
        edgecolor="white"
    )

    mean_rmse = float(sub.Stack_EN_RMSE.mean())
    std_rmse = float(sub.Stack_EN_RMSE.std(ddof=1))

    axes[1].axhline(
        mean_rmse,
        color="black",
        lw=1.5,
        ls="--",
        label=f"Three seed mean = {mean_rmse:.3f}"
    )

    axes[1].text(
        0.03, 0.92,
        f"Bars: Stack ElasticNet RMSE\n"
        f"SD = {std_rmse:.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="none")
    )

    axes[1].set_xlabel("Evaluation seed")
    axes[1].set_ylabel("RMSE, SOH %")
    axes[1].set_title(
        "B0018 LOO seed stability",
        fontweight="bold"
    )
    axes[1].legend(fontsize=8)
    add_panel_label(axes[1], "b")

    fig.suptitle(
        "Figure 12. B0018 Leave One Battery Out diagnostic",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "fig12_b0018_loo_diagnostic.png")


def figure_13_ablation(full):
    configurations = [
        "GRU",
        "LSTM",
        "XGB",
        "Weighted",
        "Stack_LR",
        "Stack_Ridge",
        "Stack_EN",
    ]

    labels = [
        "GRU",
        "LSTM",
        "XGBoost",
        "Weighted\nensemble",
        "Stack\nLR",
        "Stack\nRidge",
        "Stack\nElasticNet",
    ]

    x = np.arange(len(configurations))
    width = 0.19

    fig, ax = plt.subplots(figsize=(13, 5.7))

    for index, bid in enumerate(CELLS):
        values = []

        for configuration in configurations:
            column = f"{configuration}_RMSE"

            if column not in full.columns:
                values.append(np.nan)
            else:
                values.append(
                    float(
                        full.loc[
                            full.Battery == bid,
                            column
                        ].mean()
                    )
                )

        ax.bar(
            x + index * width,
            values,
            width,
            label=bid,
            color=PALETTE[bid],
            alpha=0.85
        )

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE, SOH percentage points")
    ax.set_xlabel("Model or ensemble configuration")

    ax.set_title(
        "Figure 13. Ablation comparison across model and fusion configurations",
        fontweight="bold"
    )

    ax.legend(
        ncol=4,
        fontsize=8.8,
        loc="upper right",
        framealpha=0.88
    )

    plt.tight_layout()
    save(fig, "fig13_ablation.png")


def main():
    print("=" * 70)
    print("Generating complete manuscript figure set")
    print("=" * 70)

    full, overall, summary, preds, rich = load_all()

    figure_1(rich)
    figure_2_workflow()
    figure_3_actual_vs_predicted(preds)
    figure_4_heatmap(summary)
    figure_5_multiseed(full)
    figure_6_residual_bars(preds)
    figure_7_residual_hist(preds)
    figure_8_learning_curves()

    loo_raw, loo_summary, loo_overall, loo_preds = load_loo()

    figure_9_loo_predictions(loo_preds, rich)
    figure_10_within_vs_loo(full, loo_overall, loo_summary)
    figure_11_b0006(loo_raw, loo_preds, rich)
    figure_12_b0018(loo_raw, loo_preds, rich)
    figure_13_ablation(full)

    print("=" * 70)
    print(f"All 13 manuscript figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
