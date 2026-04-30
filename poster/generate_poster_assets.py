from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "complete" / "benchmark_results.csv"
ASSET_DIR = ROOT / "poster" / "assets"

NAVY = "#0b2d42"
TEAL = "#087f8c"
GREEN = "#73b66b"
RED = "#d94c4c"
BLUE = "#2f6bb2"
GRAY = "#5c6670"
LIGHT_TEAL = "#e9f6f7"
LINE = "#9eb7c5"

MODEL_COLORS = {
    "PairRE": TEAL,
    "DistMult": GREEN,
    "ConvE": RED,
    "TransE": NAVY,
}


def load_summary() -> pd.DataFrame:
    results = pd.read_csv(RESULTS_PATH)
    return (
        results.groupby("model", as_index=False)
        .agg(
            avg_train_seconds=("train_seconds", "mean"),
            avg_parameters=("parameter_count", "mean"),
            avg_mrr=("mrr", "mean"),
            avg_hits_at_1=("hits@1", "mean"),
            avg_hits_at_3=("hits@3", "mean"),
            avg_hits_at_10=("hits@10", "mean"),
        )
        .sort_values("avg_mrr", ascending=False)
        .reset_index(drop=True)
    )


def save_bar_chart(
    summary: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_name: str,
    value_format: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=220)
    fig.patch.set_facecolor("white")
    colors = [MODEL_COLORS[model] for model in summary["model"]]
    bars = ax.bar(summary["model"], summary[column], color=colors, width=0.62)

    ax.set_title(title, color=NAVY, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color=NAVY, fontsize=9)
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9e2e8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ymax = summary[column].max()
    ax.set_ylim(0, ymax * 1.18)
    for bar, value in zip(bars, summary[column], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.035,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=NAVY,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(ASSET_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def save_results_table(summary: pd.DataFrame) -> None:
    table = summary.copy()
    table["avg_train_seconds"] = table["avg_train_seconds"].map(lambda value: f"{value:.2f}")
    table["avg_parameters"] = table["avg_parameters"].map(lambda value: f"{value:,.0f}")
    for column in ["avg_mrr", "avg_hits_at_1", "avg_hits_at_3", "avg_hits_at_10"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")

    table = table.rename(
        columns={
            "model": "Model",
            "avg_train_seconds": "Train s",
            "avg_parameters": "Params",
            "avg_mrr": "MRR",
            "avg_hits_at_1": "Hits@1",
            "avg_hits_at_3": "Hits@3",
            "avg_hits_at_10": "Hits@10",
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 2.4), dpi=220)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    rendered = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.65)

    for (row, _col), cell in rendered.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.9)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif table.iloc[row - 1]["Model"] == "PairRE":
            cell.set_facecolor("#eaf8f2")
            cell.get_text().set_color(TEAL)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white")

    fig.savefig(ASSET_DIR / "complete_results_table.png", bbox_inches="tight")
    plt.close(fig)


def save_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 2.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        ("Fixed\ndataset splits", "Predefined train / validation / test"),
        ("PyKEEN\ntraining", "Same seed, optimizer, epochs"),
        ("Filtered ranking\nevaluation", "MRR and Hits@k on test triples"),
        ("Model\nselection", "Quality vs. efficiency tradeoff"),
    ]

    x_positions = [0.12, 0.37, 0.62, 0.87]
    box_w = 0.17
    box_h = 0.38
    for index, ((heading, caption), x) in enumerate(zip(steps, x_positions, strict=True)):
        box = FancyBboxPatch(
            (x - box_w / 2, 0.43),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=LIGHT_TEAL,
            edgecolor=TEAL,
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(x, 0.63, heading, ha="center", va="center", color=NAVY, fontsize=9.2, fontweight="bold")
        ax.text(x, 0.26, caption, ha="center", va="center", color=GRAY, fontsize=7.5)
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + box_w / 2 + 0.02, 0.62),
                (x_positions[index + 1] - box_w / 2 - 0.02, 0.62),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color=GRAY,
            )
            ax.add_patch(arrow)

    fig.subplots_adjust(left=0.035, right=0.965, top=0.9, bottom=0.14)
    fig.savefig(ASSET_DIR / "methodology_pipeline.png")
    plt.close(fig)


def save_knowledge_graph_icon() -> None:
    graph = nx.Graph()
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    graph.add_edges_from(edges)
    positions = {
        0: (0.10, 0.65),
        1: (0.28, 0.86),
        2: (0.32, 0.42),
        3: (0.52, 0.70),
        4: (0.55, 0.26),
        5: (0.73, 0.53),
        6: (0.77, 0.13),
        7: (0.92, 0.35),
    }

    fig, ax = plt.subplots(figsize=(2.2, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color=LINE, width=2.2)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=[360, 250, 300, 380, 300, 420, 250, 340],
        node_color=["#111111", TEAL, "#111111", TEAL, "#111111", TEAL, "#111111", TEAL],
        edgecolors="#111111",
        linewidths=1.6,
    )
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    fig.savefig(ASSET_DIR / "knowledge_graph_icon.png", transparent=True)
    plt.close(fig)


def save_model_black_box_icon() -> None:
    fig, ax = plt.subplots(figsize=(2.4, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box = FancyBboxPatch(
        (0.25, 0.25),
        0.5,
        0.5,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor="#111111",
        edgecolor="#111111",
        linewidth=2,
    )
    ax.add_patch(box)

    left_arrow = FancyArrowPatch(
        (0.05, 0.5),
        (0.25, 0.5),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=3,
        color="#111111",
    )
    right_arrow = FancyArrowPatch(
        (0.75, 0.5),
        (0.95, 0.5),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=3,
        color="#111111",
    )
    ax.add_patch(left_arrow)
    ax.add_patch(right_arrow)

    for x, y in [(0.37, 0.61), (0.5, 0.61), (0.63, 0.61), (0.37, 0.39), (0.5, 0.39), (0.63, 0.39)]:
        ax.plot(x, y, marker="o", markersize=4.5, color="white")
    ax.plot([0.37, 0.5, 0.63], [0.61, 0.39, 0.61], color="white", linewidth=1.5, alpha=0.9)
    ax.plot([0.37, 0.5, 0.63], [0.39, 0.61, 0.39], color="white", linewidth=1.5, alpha=0.9)

    fig.savefig(ASSET_DIR / "model_black_box_icon.png", bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_database_icon() -> None:
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    body = Rectangle((0.2, 0.28), 0.6, 0.46, facecolor="white", edgecolor="#111111", linewidth=3)
    top = Ellipse((0.5, 0.74), 0.6, 0.18, facecolor="white", edgecolor="#111111", linewidth=3)
    bottom = Ellipse((0.5, 0.28), 0.6, 0.18, facecolor="white", edgecolor="#111111", linewidth=3)
    ax.add_patch(body)
    ax.add_patch(bottom)
    ax.add_patch(top)

    for y in [0.58, 0.43]:
        ax.plot([0.2, 0.2], [y, y + 0.01], color="#111111", linewidth=3)
        ax.plot([0.8, 0.8], [y, y + 0.01], color="#111111", linewidth=3)
        ax.add_patch(Ellipse((0.5, y), 0.6, 0.18, facecolor="none", edgecolor="#111111", linewidth=2.4))

    fig.savefig(ASSET_DIR / "database_icon.png", bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_fixed_splits_icon() -> None:
    fig, ax = plt.subplots(figsize=(2.2, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for x, y, color in (
        [
            (0.16, 0.58, "#111111"),
            (0.42, 0.58, TEAL),
            (0.68, 0.58, "#f2aa1f"),
        ]
    ):
        card = FancyBboxPatch(
            (x, y - 0.18),
            0.18,
            0.28,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            facecolor="white",
            edgecolor=color,
            linewidth=3,
        )
        ax.add_patch(card)
        ax.plot([x + 0.04, x + 0.14], [y + 0.02, y + 0.02], color=color, linewidth=2)
        ax.plot([x + 0.04, x + 0.14], [y - 0.05, y - 0.05], color=color, linewidth=2)
        ax.plot([x + 0.04, x + 0.11], [y - 0.12, y - 0.12], color=color, linewidth=2)

    ax.plot([0.17, 0.81], [0.25, 0.25], color="#111111", linewidth=2.8)
    for x in [0.25, 0.51, 0.77]:
        ax.plot([x, x], [0.25, 0.34], color="#111111", linewidth=2.8)

    fig.savefig(ASSET_DIR / "fixed_splits_icon.png", bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_filtered_ranking_icon() -> None:
    fig, ax = plt.subplots(figsize=(2.2, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for y, length in [(0.72, 0.42), (0.58, 0.34), (0.44, 0.26)]:
        ax.plot([0.15, 0.15 + length], [y, y], color="#111111", linewidth=4, solid_capstyle="round")
        ax.add_patch(Circle((0.09, y), 0.025, color=TEAL))

    lens = Circle((0.62, 0.43), 0.19, fill=False, edgecolor="#111111", linewidth=5)
    ax.add_patch(lens)
    ax.plot([0.75, 0.91], [0.29, 0.13], color="#111111", linewidth=5, solid_capstyle="round")
    ax.plot([0.49, 0.76], [0.43, 0.43], color=TEAL, linewidth=3, alpha=0.8)
    ax.plot([0.62, 0.62], [0.30, 0.56], color=TEAL, linewidth=3, alpha=0.8)

    fig.savefig(ASSET_DIR / "filtered_ranking_icon.png", bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_model_selection_icon() -> None:
    fig, ax = plt.subplots(figsize=(2.2, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    bar_data = [
        (0.16, 0.24, TEAL),
        (0.34, 0.36, GREEN),
        (0.52, 0.58, RED),
        (0.70, 0.76, "#f2aa1f"),
    ]
    for x, height, color in bar_data:
        ax.add_patch(Rectangle((x, 0.12), 0.12, height, facecolor=color, edgecolor="none"))

    ax.plot([0.1, 0.88], [0.12, 0.12], color="#111111", linewidth=3)
    ax.plot([0.1, 0.1], [0.12, 0.92], color="#111111", linewidth=3)

    # Selection badge over the chosen bar.
    badge = Circle((0.77, 0.88), 0.11, facecolor="white", edgecolor="#111111", linewidth=3)
    ax.add_patch(badge)
    ax.plot([0.72, 0.76, 0.84], [0.88, 0.82, 0.94], color=TEAL, linewidth=4, solid_capstyle="round")

    fig.savefig(ASSET_DIR / "model_selection_icon.png", bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_model_architecture_overview() -> None:
    fig, ax = plt.subplots(figsize=(5.9, 3.65), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rows = [
        ("TransE", "h + r ≈ t"),
        ("PairRE", "h ⊙ rH ≈ t ⊙ rT"),
        ("DistMult", "h ⊙ r ≈ t"),
        ("ConvE", "2D reshape + conv"),
    ]
    y_positions = [0.84, 0.60, 0.36, 0.12]

    for index, ((name, formula), y) in enumerate(zip(rows, y_positions, strict=True)):
        ax.text(0.05, y + 0.04, name, ha="left", va="center", color=TEAL, fontsize=15, fontweight="bold")
        ax.text(0.50, y + 0.04, formula, ha="center", va="center", color=NAVY, fontsize=14, style="italic")
        if index < len(rows) - 1:
            ax.plot([0.05, 0.95], [y - 0.095, y - 0.095], color=LINE, linewidth=1.4)

    # TransE: head + relation vector ~= tail.
    y = y_positions[0] - 0.04
    ax.add_patch(Circle((0.36, y), 0.034, facecolor="#b9dcf2", edgecolor=NAVY, linewidth=1.6))
    ax.add_patch(Circle((0.50, y), 0.034, facecolor="#b8e5b1", edgecolor=NAVY, linewidth=1.6))
    ax.add_patch(Circle((0.64, y), 0.034, facecolor="#b9dcf2", edgecolor=NAVY, linewidth=1.6))
    ax.text(0.43, y, "+", ha="center", va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(0.57, y, "≈", ha="center", va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(0.50, y, "r", ha="center", va="center", color=NAVY, fontsize=9, fontweight="bold")

    # PairRE: separate relation vectors scale head and tail representations.
    y = y_positions[1] - 0.04
    for x, label, fill in [(0.33, "h", "#b9dcf2"), (0.49, "rH", "#b8e5b1"), (0.61, "t", "#b9dcf2"), (0.77, "rT", "#b8e5b1")]:
        ax.add_patch(Circle((x, y), 0.030, facecolor=fill, edgecolor=NAVY, linewidth=1.4))
        ax.text(x, y, label, ha="center", va="center", color=NAVY, fontsize=7.4, fontweight="bold")

    ax.text(0.41, y, "⊙", ha="center", va="center", color=NAVY, fontsize=11, fontweight="bold")
    ax.text(0.55, y, "≈", ha="center", va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(0.69, y, "⊙", ha="center", va="center", color=NAVY, fontsize=11, fontweight="bold")

    # DistMult: elementwise interaction.
    y = y_positions[2] - 0.04
    ax.add_patch(Circle((0.36, y), 0.034, facecolor="#b9dcf2", edgecolor=NAVY, linewidth=1.6))
    ax.add_patch(Circle((0.50, y), 0.034, facecolor="#b8e5b1", edgecolor=NAVY, linewidth=1.6))
    ax.add_patch(Circle((0.64, y), 0.034, facecolor="#b9dcf2", edgecolor=NAVY, linewidth=1.6))
    ax.text(0.43, y, "⊙", ha="center", va="center", color=NAVY, fontsize=12, fontweight="bold")
    ax.text(0.57, y, "≈", ha="center", va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.text(0.50, y, "r", ha="center", va="center", color=NAVY, fontsize=9, fontweight="bold")

    # ConvE: 2D grid, convolution stack, dense output.
    y = y_positions[3] - 0.04
    grid_x, grid_y = 0.31, y - 0.036
    cell = 0.017
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((grid_x + col * cell, grid_y + row * cell), cell, cell, facecolor="#f6fbff", edgecolor=NAVY, linewidth=0.5))
    for offset in [0.0, 0.012, 0.024]:
        ax.add_patch(Rectangle((0.48 + offset, y - 0.04 + offset), 0.08, 0.08, facecolor="#cfe4f5", edgecolor=LINE, linewidth=1.0))
    ax.add_patch(Rectangle((0.66, y - 0.045), 0.025, 0.09, facecolor="#e0f2da", edgecolor=TEAL, linewidth=1.1))
    for dot_y in [y - 0.026, y, y + 0.026]:
        ax.plot(0.672, dot_y, marker="o", markersize=2.5, color=TEAL)
    ax.add_patch(FancyArrowPatch((0.40, y), (0.47, y), arrowstyle="->", mutation_scale=10, color=GRAY, linewidth=1.2))
    ax.add_patch(FancyArrowPatch((0.59, y), (0.65, y), arrowstyle="->", mutation_scale=10, color=GRAY, linewidth=1.2))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(ASSET_DIR / "model_architecture_overview.png", transparent=True)
    plt.close(fig)


def save_conclusions_tradeoff_icon() -> None:
    fig, ax = plt.subplots(figsize=(2.4, 2.0), dpi=260)
    fig.patch.set_alpha(0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Balance beam and stand.
    ax.plot([0.22, 0.78], [0.66, 0.66], color="#111111", linewidth=4, solid_capstyle="round")
    ax.plot([0.50, 0.50], [0.28, 0.66], color="#111111", linewidth=4, solid_capstyle="round")
    ax.plot([0.35, 0.65], [0.28, 0.28], color="#111111", linewidth=4, solid_capstyle="round")
    ax.add_patch(Circle((0.50, 0.66), 0.035, facecolor=TEAL, edgecolor="#111111", linewidth=2))

    # Hanging pans.
    for x, label, color in [(0.28, "MRR", TEAL), (0.72, "Cost", RED)]:
        ax.plot([x, x - 0.08], [0.66, 0.48], color="#111111", linewidth=2)
        ax.plot([x, x + 0.08], [0.66, 0.48], color="#111111", linewidth=2)
        pan = FancyBboxPatch(
            (x - 0.12, 0.40),
            0.24,
            0.08,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="white",
            edgecolor=color,
            linewidth=2.5,
        )
        ax.add_patch(pan)
        ax.text(x, 0.44, label, ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    # Selected model badge.
    badge = FancyBboxPatch(
        (0.31, 0.08),
        0.38,
        0.14,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=LIGHT_TEAL,
        edgecolor=TEAL,
        linewidth=2,
    )
    ax.add_patch(badge)
    ax.text(0.50, 0.15, "PairRE", ha="center", va="center", color=NAVY, fontsize=10, fontweight="bold")

    fig.savefig(ASSET_DIR / "conclusions_tradeoff_icon.png", bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    save_results_table(summary)
    save_bar_chart(summary, "avg_mrr", "Average MRR", "MRR", "avg_mrr_bar.png", "{:.4f}")
    save_bar_chart(
        summary,
        "avg_hits_at_10",
        "Average Hits@10",
        "Hits@10",
        "avg_hits10_bar.png",
        "{:.4f}",
    )
    save_bar_chart(
        summary,
        "avg_train_seconds",
        "Average Training Time",
        "Seconds",
        "avg_train_time_bar.png",
        "{:.0f}",
    )
    save_bar_chart(
        summary,
        "avg_parameters",
        "Average Parameter Count",
        "Parameters",
        "avg_parameter_count_bar.png",
        "{:,.0f}",
    )
    save_pipeline_diagram()
    save_knowledge_graph_icon()
    save_model_black_box_icon()
    save_database_icon()
    save_fixed_splits_icon()
    save_filtered_ranking_icon()
    save_model_selection_icon()
    save_model_architecture_overview()
    save_conclusions_tradeoff_icon()

    for path in sorted(ASSET_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
