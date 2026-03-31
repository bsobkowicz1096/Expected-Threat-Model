"""Generate model architecture diagram (training + evaluation) for thesis."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

cmap = plt.cm.inferno
COLOR_INPUT = cmap(0.80)
COLOR_EMBED = cmap(0.55)
COLOR_CORE = cmap(0.30)
COLOR_HEAD = cmap(0.45)
COLOR_OUTPUT = cmap(0.15)
COLOR_BG = "#ffffff"
COLOR_TEXT = cmap(0.05)


def draw_box(ax, cx, cy, w, h, text, color, fontsize=14, textcolor="white"):
    rect = patches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=textcolor,
            zorder=3, linespacing=1.3)


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=16), zorder=1)


def draw_training(ax):
    """Left panel: training pipeline."""
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 14)
    ax.axis("off")

    ax.text(0, 13.7, "Training", fontsize=24, fontweight="bold",
            color=COLOR_TEXT, ha="center", va="top")

    bw = 3.0
    bh = 1.0

    # Full training sequence
    y_input = 12.0
    draw_box(ax, 0, y_input, 6.5, 1.5,
             "TRAINING SEQUENCE\nTypes: Pass, Carry, Shot,\nGOAL, NO_GOAL\nPositions: x, y, x_end, y_end",
             COLOR_INPUT, fontsize=14, textcolor=COLOR_TEXT)

    # Embeddings
    y_emb = 9.8
    draw_box(ax, -1.8, y_emb, bw, bh, "TYPE\nEMBEDDING", COLOR_EMBED, fontsize=13)
    draw_box(ax, 1.8, y_emb, bw, bh, "FOURIER\nPOSITION ENC.", COLOR_EMBED, fontsize=12)
    draw_arrow(ax, -0.6, y_input - 0.75, -1.8, y_emb + 0.5)
    draw_arrow(ax, 0.6, y_input - 0.75, 1.8, y_emb + 0.5)

    # Transformer
    y_trans = 8.2
    draw_box(ax, 0, y_trans, 5.8, 0.95, "TRANSFORMER ENCODER", COLOR_CORE, fontsize=14)
    draw_arrow(ax, -1.8, y_emb - 0.5, -0.6, y_trans + 0.47)
    draw_arrow(ax, 1.8, y_emb - 0.5, 0.6, y_trans + 0.47)

    # Heads
    y_head = 6.6
    draw_box(ax, -1.8, y_head, bw, bh, "TYPE HEAD", COLOR_HEAD, fontsize=14)
    draw_box(ax, 1.8, y_head, bw, bh, "DUAL MDN\nHEAD", COLOR_HEAD, fontsize=13)
    draw_arrow(ax, -0.6, y_trans - 0.47, -1.8, y_head + 0.5)
    draw_arrow(ax, 0.6, y_trans - 0.47, 1.8, y_head + 0.5)

    # Outputs
    y_out = 5.0
    draw_box(ax, -1.8, y_out, bw, 0.8, "P(type)", COLOR_OUTPUT, fontsize=14)
    draw_box(ax, 1.8, y_out, bw, 0.8, "P(pos | type)", COLOR_OUTPUT, fontsize=13)
    draw_arrow(ax, -1.8, y_head - 0.5, -1.8, y_out + 0.4)
    draw_arrow(ax, 1.8, y_head - 0.5, 1.8, y_out + 0.4)

    # Loss
    y_loss = 3.0
    draw_box(ax, 0, y_loss, 6.5, 1.1,
             r"$\mathcal{L} = \lambda \cdot \mathcal{L}_{CE}^{type} + \mathcal{L}_{NLL}^{MDN}$",
             COLOR_OUTPUT, fontsize=18)
    draw_arrow(ax, -1.8, y_out - 0.4, -0.6, y_loss + 0.55)
    draw_arrow(ax, 1.8, y_out - 0.4, 0.6, y_loss + 0.55)

    # Best weights
    y_best = 0.8
    draw_box(ax, 0, y_best, 6.0, 1.0, "BEST WEIGHTS\n(min val loss epoch)", COLOR_CORE,
             fontsize=13)
    draw_arrow(ax, 0, y_loss - 0.55, 0, y_best + 0.5)


def draw_evaluation(ax):
    """Right panel: evaluation / xT calculation."""
    ax.set_xlim(-7, 7.5)
    ax.set_ylim(-1, 14)
    ax.axis("off")

    ax.text(0, 13.7, "Evaluation (xT)", fontsize=24, fontweight="bold",
            color=COLOR_TEXT, ha="center", va="top")

    # Context
    y_ctx = 12.0
    draw_box(ax, 0, y_ctx, 6.0, 0.95,
             "INITIAL CONTEXT",
             COLOR_INPUT, fontsize=15, textcolor=COLOR_TEXT)

    # Content positions inside the loop
    y_stype = 9.5
    y_check = 7.8
    y_spos = 6.1
    loop_x = -5.0  # feedback arrow x

    # Autoregressive loop box
    loop_top = y_stype + 1.0
    loop_bot = y_spos - 1.2
    loop_left = loop_x - 0.7
    loop_right = 5.5
    loop_rect = patches.FancyBboxPatch(
        (loop_left, loop_bot), loop_right - loop_left, loop_top - loop_bot,
        boxstyle="round,pad=0.15",
        facecolor="none", edgecolor=COLOR_EMBED,
        linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(loop_rect)
    ax.text(0, loop_top + 0.25, "autoregressive loop",
            ha="center", va="bottom", fontsize=14, color=COLOR_EMBED,
            fontstyle="italic", fontweight="bold")

    # Sample type
    draw_box(ax, 0, y_stype, 5.2, 0.9, "SAMPLE TYPE\nfrom P(type)", COLOR_HEAD, fontsize=13)
    draw_arrow(ax, 0, y_ctx - 0.47, 0, y_stype + 0.45)

    # Check terminal
    draw_box(ax, 0, y_check, 5.2, 0.8, "GOAL / NO_GOAL?", COLOR_OUTPUT, fontsize=15)
    draw_arrow(ax, 0, y_stype - 0.45, 0, y_check + 0.4)

    # YES branch
    ax.annotate("", xy=(loop_right - 0.2, y_check), xytext=(2.6, y_check),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_OUTPUT,
                                lw=1.5, mutation_scale=14), zorder=1)
    ax.text(3.95, y_check + 0.15, "yes", ha="center", va="bottom",
            fontsize=13, color=COLOR_OUTPUT, fontweight="bold")
    ax.text(loop_right + 0.15, y_check, " end\n rollout", ha="left", va="center",
            fontsize=12, color=COLOR_OUTPUT, fontstyle="italic")

    # NO branch
    draw_box(ax, 0, y_spos, 5.2, 0.9,
             "SAMPLE POSITION\nfrom MDN (pass / carry / shot)", COLOR_HEAD, fontsize=11)
    draw_arrow(ax, 0, y_check - 0.4, 0, y_spos + 0.45)
    ax.text(0.4, (y_check - 0.4 + y_spos + 0.45) / 2, "no",
            ha="left", va="center", fontsize=13, color=COLOR_OUTPUT, fontweight="bold")

    # Feedback arrow — L-shaped
    ax.plot([0, 0], [y_spos - 0.45, y_spos - 0.8], color=COLOR_EMBED, lw=1.5, zorder=1)
    ax.plot([0, loop_x], [y_spos - 0.8, y_spos - 0.8], color=COLOR_EMBED, lw=1.5, zorder=1)
    ax.plot([loop_x, loop_x], [y_spos - 0.8, y_stype + 0.8], color=COLOR_EMBED, lw=1.5, zorder=1)
    ax.plot([loop_x, 0], [y_stype + 0.8, y_stype + 0.8], color=COLOR_EMBED, lw=1.5, zorder=1)
    ax.annotate("", xy=(0, y_stype + 0.45), xytext=(0, y_stype + 0.8),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_EMBED,
                                lw=1.5, mutation_scale=14), zorder=1)
    ax.text(loop_x + 0.2, (y_spos + y_stype) / 2, "append\ntoken",
            ha="left", va="center", fontsize=12, color=COLOR_EMBED,
            fontstyle="italic", fontweight="bold")

    # "Repeat N times" as label under the dashed frame
    ax.text(0, loop_bot - 0.3, "× N rollouts (Monte Carlo)",
            ha="center", va="top", fontsize=13, color=COLOR_EMBED,
            fontstyle="italic", fontweight="bold")

    # xT formula
    y_xt = 3.0
    draw_box(ax, 0, y_xt, 5.5, 1.0,
             r"$xT = \frac{\#\ GOAL}{N}$",
             COLOR_OUTPUT, fontsize=20)
    draw_arrow(ax, 0, loop_bot - 0.7, 0, y_xt + 0.5)

    # Metrics
    y_met = 0.8
    draw_box(ax, 0, y_met, 6.5, 0.9, "ROC-AUC, Brier, Separation",
             COLOR_CORE, fontsize=14)
    ax.text(3.25, y_met + 0.45 + 0.18, "metrics", ha="right", va="bottom",
            fontsize=11, color=COLOR_CORE, fontstyle="italic", fontweight="bold")
    draw_arrow(ax, 0, y_xt - 0.5, 0, y_met + 0.45)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    fig.patch.set_facecolor(COLOR_BG)

    draw_training(ax1)
    draw_evaluation(ax2)

    fig.tight_layout(w_pad=2)
    fig.savefig("thesis_figures/viz_architecture.png", dpi=200,
                bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print("Saved viz_architecture.png")


if __name__ == "__main__":
    main()
