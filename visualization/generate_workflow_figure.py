
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def box(ax, x, y, w, h, title, body, face, text_color="white", title_size=9.0, body_size=8.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="white",
        facecolor=face,
        zorder=3
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h*0.64, title,
            ha="center", va="center", color=text_color,
            fontsize=title_size, fontweight="bold", zorder=4)
    ax.text(x + w/2, y + h*0.28, body,
            ha="center", va="center", color=text_color,
            fontsize=body_size, linespacing=1.25, zorder=4)
    return patch

def arrow(ax, x1, y1, x2, y2, rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color="#4B5563",
        connectionstyle=f"arc3,rad={rad}",
        zorder=2
    ))

def build():
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    navy = "#203D70"
    blue = "#2774B9"
    teal = "#319B95"
    grey = "#555A60"
    green = "#1F5C3A"
    orange = "#F3A052"
    purple = "#752A84"
    light_blue = "#4A9DDA"
    light_green = "#35A979"
    magenta = "#D11879"
    brown = "#B65E16"

    ax.text(6.75, 7.73, "Experimental Workflow and Model Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=navy)

    # Top data processing row
    box(ax, 0.35, 6.25, 1.75, 0.95, "NASA MAT Files", "B0005 to B0018", navy)
    box(ax, 2.45, 6.25, 1.75, 0.95, "Discharge\nExtraction", "V, I, T, Capacity", blue, title_size=8.6)
    box(ax, 4.55, 6.25, 1.85, 0.95, "20 Feature\nExtraction", "per cycle", teal, title_size=8.6)
    box(ax, 6.75, 6.25, 1.65, 0.95, "MinMax Scaler", "train only", grey)
    box(ax, 8.75, 6.25, 2.05, 0.95, "Sliding Window", "SEQ_LEN = 32\nXGB lag = 5", navy)

    arrow(ax, 2.10, 6.73, 2.45, 6.73)
    arrow(ax, 4.20, 6.73, 4.55, 6.73)
    arrow(ax, 6.40, 6.73, 6.75, 6.73)
    arrow(ax, 8.40, 6.73, 8.75, 6.73)

    ax.text(5.60, 6.06, "Data ingestion and preprocessing",
            ha="center", fontsize=8.5, color="#555555", style="italic")

    # Chronological split row
    ax.text(0.35, 5.35, "Chronological Split",
            fontsize=10.2, fontweight="bold", color=navy, va="center")

    box(ax, 0.35, 4.60, 1.95, 0.82, "Chronological Split", "60 / 10 / 15 / 15 %", green, body_size=8)
    box(ax, 2.55, 4.60, 1.65, 0.82, "Base", "60 %", navy)
    box(ax, 4.45, 4.60, 1.35, 0.82, "EV", "10 %", orange, text_color="white")
    box(ax, 6.05, 4.60, 1.45, 0.82, "Meta", "15 %", light_green)
    box(ax, 7.75, 4.60, 1.35, 0.82, "Test", "15 %", purple)

    arrow(ax, 2.30, 5.01, 2.55, 5.01)
    arrow(ax, 4.20, 5.01, 4.45, 5.01)
    arrow(ax, 5.80, 5.01, 6.05, 5.01)
    arrow(ax, 7.50, 5.01, 7.75, 5.01)

    ax.text(5.05, 4.28,
            "Strictly chronological, no future data, leakage free",
            ha="center", fontsize=8.3, color="#555555", style="italic")

    # Base learners
    box(ax, 3.35, 3.00, 1.85, 0.85, "GRU", "2 layers, 64 units\ndropout = 0.20", light_blue, body_size=7.6)
    box(ax, 5.70, 3.00, 1.85, 0.85, "LSTM", "2 layers, 64 units\ndropout = 0.20", light_green, body_size=7.6)
    box(ax, 8.05, 3.00, 1.85, 0.85, "XGBoost", "300 trees, depth = 4\nlag features", magenta, body_size=7.6)

    # Base training route: below the split row, then branch to the learners
    arrow(ax, 3.38, 4.60, 3.38, 4.05)
    ax.plot([3.38, 8.98], [4.05, 4.05], color="#4B5563", linewidth=1.2, zorder=1)
    ax.text(3.52, 4.12, "training data", fontsize=7.3, color="#555555", style="italic")
    arrow(ax, 4.28, 4.05, 4.28, 3.86)
    arrow(ax, 6.63, 4.05, 6.63, 3.86)
    arrow(ax, 8.98, 4.05, 8.98, 3.86)

    # Sliding window route: right side, then a separate bus above the model row
    arrow(ax, 9.78, 6.25, 11.35, 6.25)
    arrow(ax, 11.35, 6.25, 11.35, 4.28)
    ax.plot([4.28, 11.35], [4.28, 4.28], color="#4B5563", linewidth=1.2, zorder=1)
    ax.text(9.85, 4.35, "sequence input", fontsize=7.3, color="#555555", style="italic")
    arrow(ax, 4.28, 4.28, 4.28, 3.86)
    arrow(ax, 6.63, 4.28, 6.63, 3.86)
    arrow(ax, 8.98, 4.28, 8.98, 3.86)

    ax.text(6.75, 2.78,
            "Base learners use the sliding window input; training uses the base block and early stopping uses the EV block",
            ha="center", fontsize=7.8, color="#555555", style="italic")

    # Meta learner
    box(ax, 4.70, 1.70, 3.90, 0.82,
        "Meta predictions to\nElasticNet stacking",
        "alpha = 0.05, l1 ratio = 0.50",
        brown, body_size=7.8, title_size=8.7)

    arrow(ax, 4.28, 3.00, 5.25, 2.52, rad=0.02)
    arrow(ax, 6.63, 3.00, 6.65, 2.52, rad=0.00)
    arrow(ax, 8.98, 3.00, 7.95, 2.52, rad=-0.02)

    # Test evaluation
    box(ax, 9.35, 1.70, 3.45, 0.82,
        "Test set evaluation",
        "never seen during training",
        navy, body_size=8.5)

    # Test route follows the right edge and does not cross model boxes
    arrow(ax, 8.42, 5.01, 12.10, 4.20, rad=-0.08)
    arrow(ax, 12.10, 4.20, 12.10, 2.52)

    ax.text(6.75, 0.78,
            "LOO: held out battery is the test set; training cells provide base, EV, and meta blocks",
            ha="center", fontsize=8.2, color="#555555", style="italic")

    ax.text(6.75, 0.38,
            "Within battery: 60 / 10 / 15 / 15 %    |    LOO training cells: 70 / 15 / 15 %",
            ha="center", fontsize=8.0, color="#555555")

    fig.savefig(OUT / "fig_02_workflow_corrected.png",
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    build()
