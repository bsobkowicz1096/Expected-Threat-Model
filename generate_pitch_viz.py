"""Generate pitch visualization of example sequence for thesis."""

import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np

# Inferno palette (matching flow diagram)
cmap = plt.cm.inferno
COLOR_PASS = cmap(0.55)
COLOR_CARRY = cmap(0.30)
COLOR_SHOT = cmap(0.80)
COLOR_BG = "#ffffff"

# Sequence data — carry 9 goes deeper into the box
events = [
    (1,  "Pass",  (17.2, 27.5), (35.8, 74.3)),
    (2,  "Carry", (35.8, 74.3), (42.1, 68.7)),
    (3,  "Pass",  (42.1, 68.7), (64.5, 5.2)),
    (4,  "Carry", (64.5, 5.2),  (68.9, 4.3)),
    (5,  "Pass",  (68.9, 4.3),  (78.2, 15.6)),
    (6,  "Carry", (78.2, 15.6), (82.4, 18.9)),
    (7,  "Pass",  (82.4, 18.9), (98.3, 35.4)),
    (8,  "Pass",  (98.3, 35.4), (105.6, 13.8)),
    (9,  "Carry", (105.6, 13.8),(113.5, 27.0)),
    (10, "Pass",  (113.5, 27.0),(99.5, 55.3)),
    (11, "Shot",  (99.5, 55.3), (120.0, 37.2)),
]


def color_for_type(t):
    if t == "Pass":  return COLOR_PASS
    if t == "Carry": return COLOR_CARRY
    if t == "Shot":  return COLOR_SHOT
    return "gray"


def main():
    pitch = Pitch(pitch_type="custom", pitch_length=120, pitch_width=80,
                  pitch_color="#3a7a33", line_color="white", linewidth=1.5,
                  goal_type="box")
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.patch.set_facecolor(COLOR_BG)

    for ev in events:
        idx, etype, start, end = ev
        col = color_for_type(etype)
        sx, sy = start
        ex, ey = end
        mx, my = (sx + ex) / 2, (sy + ey) / 2

        if etype == "Carry":
            ax.plot([sx, ex], [sy, ey], color=col, lw=2.2,
                    linestyle="--", zorder=3)
            ax.plot(sx, sy, 'o', color=col, ms=5, zorder=4)
        elif etype == "Shot":
            pitch.arrows(sx, sy, ex, ey, ax=ax, color=col,
                         width=2.5, headwidth=6, headlength=4, zorder=3)
        else:
            pitch.arrows(sx, sy, ex, ey, ax=ax, color=col,
                         width=2, headwidth=5, headlength=3.5, zorder=3)

        # Label at midpoint of segment
        ax.annotate(str(idx), xy=(mx, my),
                    fontsize=11, fontweight="bold", color="white",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=col,
                              edgecolor=col, alpha=0.85),
                    zorder=5)

    # End point of shot (goal)
    ex, ey = events[-1][3]
    ax.plot(ex, ey, '*', color=COLOR_SHOT, ms=12, zorder=5,
            markeredgecolor="white", markeredgewidth=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_PASS, lw=2.2, marker='o', ms=5,
               label="Pass"),
        Line2D([0], [0], color=COLOR_CARRY, lw=2.2, linestyle="--",
               marker='o', ms=5, label="Carry"),
        Line2D([0], [0], color=COLOR_SHOT, lw=2.5, marker='o', ms=5,
               label="Shot"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left",
                    fontsize=10, framealpha=0.9, edgecolor="#cccccc",
                    fancybox=True, prop={"weight": "bold"})
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig("thesis_figures/viz_pitch_sequence.png", dpi=200,
                bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print("Saved viz_pitch_sequence.png")


if __name__ == "__main__":
    main()
