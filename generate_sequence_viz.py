"""Generate pass+carry sequence visualizations for thesis."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import matplotlib

# Use inferno palette - pick distinct points along the colormap
cmap = plt.cm.inferno
COLOR_PASS = cmap(0.55)    # warm orange
COLOR_CARRY = cmap(0.30)   # dark magenta/purple
COLOR_SHOT = cmap(0.80)    # bright yellow
COLOR_GOAL = cmap(0.15)    # deep purple/dark
COLOR_HEADER = cmap(0.05)  # near-black for table header
COLOR_BG = "#ffffff"       # white background

# Example sequence data
events = [
    (1,  "Pass",  "(17.2, 27.5)", "(35.8, 74.3)"),
    (2,  "Carry", "(35.8, 74.3)", "(42.1, 68.7)"),
    (3,  "Pass",  "(42.1, 68.7)", "(64.5, 5.2)"),
    (4,  "Carry", "(64.5, 5.2)",  "(68.9, 4.3)"),
    (5,  "Pass",  "(68.9, 4.3)",  "(78.2, 15.6)"),
    (6,  "Carry", "(78.2, 15.6)", "(82.4, 18.9)"),
    (7,  "Pass",  "(82.4, 18.9)", "(98.3, 35.4)"),
    (8,  "Pass",  "(98.3, 35.4)", "(105.6, 13.8)"),
    (9,  "Carry", "(105.6, 13.8)","(111.2, 17.4)"),
    (10, "Pass",  "(111.2, 17.4)","(99.5, 55.3)"),
    (11, "Shot",  "(99.5, 55.3)", "(120.0, 37.2)"),
]

def color_for_type(t):
    if t == "Pass":  return COLOR_PASS
    if t == "Carry": return COLOR_CARRY
    if t == "Shot":  return COLOR_SHOT
    if t == "Goal":  return COLOR_GOAL
    return "gray"

# ── Figure 1: Raw sequence table ──────────────────────────────────────
def draw_table():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.axis("off")

    cols = ["N", "Type", "Start", "End"]
    col_widths = [0.08, 0.22, 0.35, 0.35]

    table = ax.table(
        cellText=[(str(e[0]), e[1], e[2], e[3]) for e in events],
        colLabels=cols,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(COLOR_HEADER)
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        else:
            event_type = events[row - 1][1]
            base_color = to_rgba(color_for_type(event_type))
            light = (*base_color[:3], 0.12)
            cell.set_facecolor(light)
            if col == 1:
                cell.set_text_props(fontweight="bold", color=color_for_type(event_type))

    ax.set_title("Raw event sequence (pass + carry)", fontsize=13,
                 fontweight="bold", pad=15, color=COLOR_HEADER)
    fig.tight_layout()
    fig.savefig("viz_sequence_table.png", dpi=200, bbox_inches="tight",
                facecolor=COLOR_BG)
    plt.close(fig)
    print("Saved viz_sequence_table.png")


# ── Figure 2: Flow diagram (pass+carry model view) ───────────────────
def draw_flow():
    # Build the model's view: passes + carries + shot + goal
    model_events = []
    for e in events:
        model_events.append(e)
    # Add goal as terminal
    model_events.append((12, "Goal", "-", "-"))

    n = len(model_events)
    box_h = 0.55
    gap = 0.35
    total_h = n * box_h + (n - 1) * gap
    fig_h = max(total_h * 0.65 + 1.5, 8)

    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, total_h + 1.5)
    ax.axis("off")

    ax.set_title("Model input sequence", fontsize=13,
                 fontweight="bold", pad=8, color=COLOR_HEADER)

    # Column header
    header_y = total_h + 0.6
    header_cols = [
        (0.75, "Index"),
        (2.1, "Type"),
        (4.1, "Start"),
        (5.7, "End"),
    ]
    for hx, hlabel in header_cols:
        ax.text(hx, header_y, hlabel, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#888888",
                fontstyle="italic")

    for i, ev in enumerate(model_events):
        y = total_h - i * (box_h + gap)
        etype = ev[1]
        col = color_for_type(etype)

        # Box
        rect = patches.FancyBboxPatch(
            (0.3, y - box_h / 2), 5.8, box_h,
            boxstyle="round,pad=0.08",
            facecolor="white",
            edgecolor=col,
            linewidth=2.5,
        )
        ax.add_patch(rect)

        # Left accent bar
        accent = patches.FancyBboxPatch(
            (0.3, y - box_h / 2), 0.15, box_h,
            boxstyle="round,pad=0.02",
            facecolor=col,
            edgecolor=col,
            linewidth=0,
        )
        ax.add_patch(accent)

        # Text
        idx_str = str(ev[0])
        type_str = ev[1].upper()
        start_str = ev[2]
        end_str = ev[3]

        ax.text(0.75, y, idx_str, ha="center", va="center",
                fontsize=10, fontweight="bold", color=col)
        ax.text(1.3, y, "|", ha="center", va="center",
                fontsize=10, color="#cccccc")
        ax.text(2.1, y, type_str, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.text(3.2, y, "|", ha="center", va="center",
                fontsize=10, color="#cccccc")
        ax.text(4.1, y, start_str, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#555555")
        ax.text(5.0, y, "|", ha="center", va="center",
                fontsize=10, color="#cccccc")
        ax.text(5.7, y, end_str, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#555555")

        # Arrow to next box
        if i < n - 1:
            arrow_y = y - box_h / 2 - gap / 2
            ax.annotate("", xy=(3.2, arrow_y - gap * 0.15),
                        xytext=(3.2, arrow_y + gap * 0.15),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.8, mutation_scale=14))

    # Legend
    legend_y = -0.2
    legend_items = [("Pass", COLOR_PASS), ("Carry", COLOR_CARRY),
                    ("Shot", COLOR_SHOT), ("Goal", COLOR_GOAL)]
    for j, (label, c) in enumerate(legend_items):
        x = 0.8 + j * 1.5
        rect = patches.FancyBboxPatch(
            (x, legend_y - 0.12), 0.3, 0.24,
            boxstyle="round,pad=0.03", facecolor=c, edgecolor=c)
        ax.add_patch(rect)
        ax.text(x + 0.45, legend_y, label, va="center", fontsize=9,
                fontweight="bold", color="#333333")

    fig.tight_layout()
    fig.savefig("thesis_figures/viz_sequence_flow.png", dpi=200, bbox_inches="tight",
                facecolor=COLOR_BG)
    plt.close(fig)
    print("Saved viz_sequence_flow.png")


if __name__ == "__main__":
    draw_flow()
    print("Done.")
