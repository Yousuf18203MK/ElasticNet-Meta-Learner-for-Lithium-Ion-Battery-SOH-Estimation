

from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import RESULTS_DIR, FIGURES_DIR, BATTERIES
from src.feature_engineering import extract_rich_features

warnings.filterwarnings("ignore")

CELLS = list(BATTERIES.keys())
LOO_DIR = RESULTS_DIR / "loo"
SEEDS = [42, 123, 2024]

PALETTE = {
    "B0005": "#2166ac",
    "B0006": "#d6604d",
    "B0007": "#1a9850",
    "B0018": "#7b2d8b",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
})


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {name}")


def find_column(df, candidates, required=True):
    normalized = {
        str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", ""): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            "Could not identify required column. "
            f"Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_prediction_frame(df):
    """
    Convert different LOO prediction schemas into one internal schema.

    Required internal columns:
        Cycle
        Actual
        GRU
        LSTM
        XGB
        Weighted
        Stack_LR
        Stack_Ridge
        Stack_EN
    """

    cycle = find_column(
        df,
        ["Cycle", "cycle", "Discharge_Cycle", "discharge_cycle", "index"]
    )

    actual = find_column(
        df,
        [
            "Actual_SOH",
            "actual_soh",
            "Actual",
            "actual",
            "SOH",
            "soh",
            "y_true",
            "target",
        ]
    )

    output = pd.DataFrame()
    output["Cycle"] = pd.to_numeric(df[cycle], errors="coerce")
    output["Actual"] = pd.to_numeric(df[actual], errors="coerce")

    model_candidates = {
        "GRU": ["GRU", "GRU_Pred", "GRU_Prediction", "gru_prediction"],
        "LSTM": ["LSTM", "LSTM_Pred", "LSTM_Prediction", "lstm_prediction"],
        "XGB": [
            "XGB",
            "XGBoost",
            "XGB_Pred",
            "XGB_Prediction",
            "xgb_prediction",
        ],
        "Weighted": [
            "Weighted",
            "Weighted_Ensemble",
            "Weighted_Pred",
            "Weighted_Prediction",
        ],
        "Stack_LR": [
            "Stack_LR",
            "StackLR",
            "Stack_LR_Pred",
            "Stack_LR_Prediction",
        ],
        "Stack_Ridge": [
            "Stack_Ridge",
            "StackRidge",
            "Stack_Ridge_Pred",
            "Stack_Ridge_Prediction",
        ],
        "Stack_EN": [
            "Stack_EN",
            "Stack_EN_Pred",
            "Stack_EN_Prediction",
            "ElasticNet",
            "ElasticNet_Pred",
            "EN",
            "EN_Pred",
            "Prediction",
        ],
    }

    for internal, candidates in model_candidates.items():
        col = find_column(df, candidates, required=False)

        if col is not None:
            output[internal] = pd.to_numeric(
                df[col], errors="coerce"
            )

    required_models = ["GRU", "LSTM", "XGB", "Stack_EN"]

    missing = [m for m in required_models if m not in output.columns]

    if missing:
        raise ValueError(
            "LOO prediction file does not contain the required model columns: "
            f"{missing}. "
            f"Available columns are: {list(df.columns)}"
        )

    output = output.dropna(subset=["Cycle", "Actual", "Stack_EN"])
    output = output.sort_values("Cycle").reset_index(drop=True)

    return output


def load_loo_prediction_files():
    """
    Load LOO predictions from CSV files.

    The function first looks for the conventional per battery files.
    It then falls back to a combined CSV if available.
    """

    predictions = {}

    for battery in CELLS:
        candidates = [
            LOO_DIR / f"{battery}_loo_predictions_s42.csv",
            LOO_DIR / f"{battery}_predictions_s42.csv",
            LOO_DIR / f"{battery}_loo_s42.csv",
        ]

        found = next((p for p in candidates if p.exists()), None)

        if found is not None:
            raw = pd.read_csv(found)
            predictions[battery] = normalize_prediction_frame(raw)
            print(f"Loaded {found.name}")
            continue

    if len(predictions) == len(CELLS):
        return predictions

    pickle_candidates = [
        LOO_DIR / "loo_preds_s42.pkl",
        LOO_DIR / "loo_predictions_s42.pkl",
    ]

    pickle_path = next(
        (p for p in pickle_candidates if p.exists()),
        None
    )

    if pickle_path is not None:
        with open(pickle_path, "rb") as f:
            obj = pickle.load(f)

        for battery in CELLS:
            if battery not in obj:
                continue

            value = obj[battery]

            if isinstance(value, pd.DataFrame):
                raw = value
            else:
                raw = pd.DataFrame(value)

            predictions[battery] = normalize_prediction_frame(raw)

    if len(predictions) != len(CELLS):
        missing = [b for b in CELLS if b not in predictions]

        raise FileNotFoundError(
            "Could not load LOO predictions for: "
            f"{missing}. "
            "Expected per battery CSV files or loo_preds_s42.pkl."
        )

    return predictions


def load_loo_tables():
    raw_path = LOO_DIR / "loo_results_raw.csv"
    summary_path = LOO_DIR / "loo_summary_per_battery.csv"
    overall_path = LOO_DIR / "loo_overall.csv"

    for path in [raw_path, summary_path, overall_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing LOO result file: {path}")

    return (
        pd.read_csv(raw_path),
        pd.read_csv(summary_path),
        pd.read_csv(overall_path),
    )


def load_rich_features():
    rich = {}

    for battery in CELLS:
        rich_path = RESULTS_DIR / f"{battery}_rich.csv"

        if rich_path.exists():
            rich[battery] = pd.read_csv(rich_path)
            continue

        mat_path = Path(
            __file__
        ).resolve().parents[1] / "data" / BATTERIES[battery]

        if mat_path.exists():
            rich[battery] = extract_rich_features(mat_path, battery)

    return rich


def add_panel_label(ax, label):
    ax.text(
        0.015,
        0.965,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def figure_1_loo_overall(predictions):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, battery in zip(axes, CELLS):
        p = predictions[battery]

        ax.plot(
            p.Cycle,
            p.Actual,
            color="black",
            linewidth=2.3,
            label="Actual LOO test SOH",
        )

        ax.plot(
            p.Cycle,
            p.GRU,
            "--",
            color="#2166ac",
            linewidth=1.3,
            label="GRU",
        )

        ax.plot(
            p.Cycle,
            p.LSTM,
            "--",
            color="#4dac26",
            linewidth=1.3,
            label="LSTM",
        )

        ax.plot(
            p.Cycle,
            p.XGB,
            ":",
            color="#d01c8b",
            linewidth=1.4,
            label="XGBoost",
        )

        ax.plot(
            p.Cycle,
            p.Stack_EN,
            color="#bf5b17",
            linewidth=2.2,
            label="Stack ElasticNet",
        )

        rmse = np.sqrt(
            np.mean((p.Actual.values - p.Stack_EN.values) ** 2)
        )

        ax.set_title(
            f"{battery} | LOO RMSE = {rmse:.3f}",
            fontweight="bold",
        )

        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("SOH (%)")

    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    add_panel_label(axes[2], "c")
    add_panel_label(axes[3], "d")

    axes[0].legend(
        loc="lower left",
        ncol=2,
        fontsize=7.6,
        framealpha=0.9,
    )

    fig.suptitle(
        "Figure 1. Leave One Battery Out actual versus predicted SOH",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    save(fig, "loo_fig01_actual_vs_predicted.png")


def figure_2_model_comparison(loo_overall):
    models = [
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
        "Stack LR",
        "Stack Ridge",
        "Stack ElasticNet",
    ]

    values = []

    for model in models:
        row = loo_overall[
            loo_overall["Model"].astype(str).str.strip() == model
        ]

        values.append(
            float(row["RMSE_mean"].iloc[0])
            if len(row)
            else np.nan
        )

    fig, ax = plt.subplots(figsize=(10, 5.7))

    bars = ax.bar(
        labels,
        values,
        color=[
            "#4d9de0",
            "#3bb273",
            "#d01c8b",
            "#8073ac",
            "#e08214",
            "#35978f",
            "#bf5b17",
        ],
        alpha=0.86,
        edgecolor="white",
    )

    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.06,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
            )

    ax.set_ylabel("Mean RMSE, SOH percentage points")
    ax.set_xlabel("Model or ensemble configuration")
    ax.set_title(
        "Figure 2. Cross battery LOO model comparison",
        fontweight="bold",
    )

    ax.text(
        0.01,
        0.97,
        "Bars represent mean RMSE across four held out batteries and three seeds",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(
            facecolor="white",
            alpha=0.86,
            edgecolor="none",
        ),
    )

    plt.tight_layout()
    save(fig, "loo_fig02_model_comparison.png")


def figure_3_battery_comparison(loo_summary):
    fig, ax = plt.subplots(figsize=(9, 5.7))

    x = np.arange(len(CELLS))
    width = 0.24

    for idx, model in enumerate(["GRU", "LSTM", "Weighted", "Stack_EN"]):
        values = []

        for battery in CELLS:
            rows = loo_summary[
                (loo_summary["Test_Battery"] == battery)
                & (loo_summary["Model"] == model)
            ]

            values.append(
                float(rows["RMSE_mean"].iloc[0])
                if len(rows)
                else np.nan
            )

        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=model,
            alpha=0.86,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_xlabel("Held out battery")
    ax.set_ylabel("Mean RMSE, SOH percentage points")
    ax.set_title(
        "Figure 3. LOO RMSE by held out battery",
        fontweight="bold",
    )

    ax.legend(
        title="Model",
        fontsize=8.5,
        framealpha=0.9,
    )

    ax.text(
        0.01,
        0.97,
        "Each bar summarizes the three evaluation seeds",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(
            facecolor="white",
            alpha=0.86,
            edgecolor="none",
        ),
    )

    plt.tight_layout()
    save(fig, "loo_fig03_rmse_by_battery.png")


def figure_4_b0006_diagnostic(predictions, loo_raw):
    battery = "B0006"
    p = predictions[battery]

    sub = loo_raw[
        loo_raw["Test_Battery"] == battery
    ].copy()

    fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.15, 1],
        hspace=0.42,
        wspace=0.34,
    )

    ax0 = fig.add_subplot(gs[0, :])

    ax0.plot(
        p.Cycle,
        p.Actual,
        color="#d6604d",
        linewidth=2.4,
        label="Actual LOO test SOH",
    )

    ax0.plot(
        p.Cycle,
        p.Stack_EN,
        color="#bf5b17",
        linewidth=2.2,
        label="Stack ElasticNet",
    )

    ax0.plot(
        p.Cycle,
        p.GRU,
        "--",
        color="#2166ac",
        linewidth=1.3,
        label="GRU",
    )

    ax0.plot(
        p.Cycle,
        p.LSTM,
        "--",
        color="#4dac26",
        linewidth=1.3,
        label="LSTM",
    )

    rmse = np.sqrt(
        np.mean((p.Actual.values - p.Stack_EN.values) ** 2)
    )

    ax0.set_title(
        f"B0006 LOO, seed 42 | Stack ElasticNet RMSE = {rmse:.3f}",
        fontweight="bold",
    )

    ax0.set_xlabel("Discharge cycle")
    ax0.set_ylabel("SOH (%)")
    ax0.legend(
        ncol=2,
        fontsize=8,
        framealpha=0.9,
    )

    add_panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1, 0])

    ax1.bar(
        sub["Seed"].astype(str),
        sub["Stack_EN_RMSE"],
        color="#d6604d",
        alpha=0.85,
        edgecolor="white",
    )

    mean_value = float(sub["Stack_EN_RMSE"].mean())
    std_value = float(sub["Stack_EN_RMSE"].std(ddof=1))

    ax1.axhline(
        mean_value,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Three seed mean = {mean_value:.3f}",
    )

    ax1.set_xlabel("Evaluation seed")
    ax1.set_ylabel("RMSE, SOH %")
    ax1.set_title(
        "B0006 seed stability",
        fontweight="bold",
    )

    ax1.text(
        0.03,
        0.93,
        f"Bars: Stack ElasticNet RMSE\n"
        f"SD across seeds = {std_value:.3f}",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            alpha=0.88,
            edgecolor="none",
        ),
    )

    ax1.legend(fontsize=7.8)
    add_panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, 1])

    residual = p.Stack_EN.values - p.Actual.values

    ax2.scatter(
        p.Actual,
        residual,
        color="#d6604d",
        s=32,
        alpha=0.76,
    )

    ax2.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="Zero residual",
    )

    ax2.set_xlabel("Actual SOH (%)")
    ax2.set_ylabel("Residual, predicted minus actual SOH (%)")
    ax2.set_title(
        "B0006 residual versus actual SOH",
        fontweight="bold",
    )

    ax2.legend(fontsize=7.8)
    add_panel_label(ax2, "c")

    fig.suptitle(
        "Figure 4. B0006 LOO diagnostic",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    save(fig, "loo_fig04_b0006_diagnostic.png")


def figure_5_b0018_diagnostic(predictions, loo_raw):
    battery = "B0018"
    p = predictions[battery]

    sub = loo_raw[
        loo_raw["Test_Battery"] == battery
    ].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
    )

    axes[0].plot(
        p.Cycle,
        p.Actual,
        color="#7b2d8b",
        linewidth=2.4,
        label="Actual LOO test SOH",
    )

    axes[0].plot(
        p.Cycle,
        p.Stack_EN,
        color="#bf5b17",
        linewidth=2.2,
        label="Stack ElasticNet",
    )

    axes[0].plot(
        p.Cycle,
        p.GRU,
        "--",
        color="#2166ac",
        linewidth=1.3,
        label="GRU",
    )

    axes[0].plot(
        p.Cycle,
        p.LSTM,
        "--",
        color="#4dac26",
        linewidth=1.3,
        label="LSTM",
    )

    axes[0].set_xlabel("Discharge cycle")
    axes[0].set_ylabel("SOH (%)")
    axes[0].set_title(
        "B0018 LOO prediction trajectory",
        fontweight="bold",
    )
    axes[0].legend(fontsize=7.8)
    add_panel_label(axes[0], "a")

    axes[1].bar(
        sub["Seed"].astype(str),
        sub["Stack_EN_RMSE"],
        color="#7b2d8b",
        alpha=0.85,
        edgecolor="white",
    )

    mean_value = float(sub["Stack_EN_RMSE"].mean())
    std_value = float(sub["Stack_EN_RMSE"].std(ddof=1))

    axes[1].axhline(
        mean_value,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Three seed mean = {mean_value:.3f}",
    )

    axes[1].set_xlabel("Evaluation seed")
    axes[1].set_ylabel("RMSE, SOH %")
    axes[1].set_title(
        "B0018 seed stability",
        fontweight="bold",
    )

    axes[1].text(
        0.03,
        0.93,
        f"Bars: Stack ElasticNet RMSE\n"
        f"SD across seeds = {std_value:.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            alpha=0.88,
            edgecolor="none",
        ),
    )

    axes[1].legend(fontsize=7.8)
    add_panel_label(axes[1], "b")

    fig.suptitle(
        "Figure 5. B0018 LOO diagnostic",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    save(fig, "loo_fig05_b0018_diagnostic.png")


def figure_6_residual_distributions(predictions):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13, 4.5),
    )

    for ax, battery in zip(axes, CELLS):
        p = predictions[battery]

        residual = p.Stack_EN.values - p.Actual.values

        mean_value = float(np.mean(residual))
        std_value = float(np.std(residual, ddof=1))

        ax.hist(
            residual,
            bins=10,
            color=PALETTE[battery],
            alpha=0.82,
            edgecolor="white",
        )

        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="Zero residual",
        )

        ax.axvline(
            mean_value,
            color="red",
            linewidth=1.4,
            label=f"Mean = {mean_value:.3f}",
        )

        ax.set_title(
            f"{battery} | SD = {std_value:.3f}",
            fontweight="bold",
        )

        ax.set_xlabel(
            "Residual, predicted minus actual SOH (%)"
        )

        if battery == CELLS[0]:
            ax.set_ylabel("Count")

        ax.legend(
            fontsize=7.4,
            framealpha=0.88,
        )

    fig.suptitle(
        "Figure 6. LOO residual distributions for Stack ElasticNet",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    save(fig, "loo_fig06_residual_distributions.png")


def main():
    print("=" * 70)
    print("Generating complete LOO manuscript figure set")
    print("=" * 70)

    predictions = load_loo_prediction_files()
    loo_raw, loo_summary, loo_overall = load_loo_tables()

    print(
        f"Loaded LOO predictions for: {', '.join(predictions.keys())}"
    )

    figure_1_loo_overall(predictions)
    figure_2_model_comparison(loo_overall)
    figure_3_battery_comparison(loo_summary)
    figure_4_b0006_diagnostic(predictions, loo_raw)
    figure_5_b0018_diagnostic(predictions, loo_raw)
    figure_6_residual_distributions(predictions)

    print("=" * 70)
    print(
        f"Complete LOO figure set saved to: {FIGURES_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
