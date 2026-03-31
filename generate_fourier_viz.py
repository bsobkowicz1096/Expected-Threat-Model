"""Generate Fourier encoding visualization for thesis."""

import matplotlib.pyplot as plt
import numpy as np

# Inferno palette
cmap = plt.cm.inferno
COLOR_LOW = cmap(0.30)     # dark purple — low freq
COLOR_MID = cmap(0.55)     # warm orange — mid freq
COLOR_HIGH = cmap(0.80)    # bright yellow — high freq
COLOR_BG = "#ffffff"
COLOR_TEXT = cmap(0.05)    # near-black

x = np.linspace(0.0, 1.0, 2000)
freqs = [(1, "f=1 (niska)", COLOR_LOW),
         (16, "f=16 (średnia)", COLOR_MID),
         (128, "f=128 (wysoka)", COLOR_HIGH)]

sample_points = [0.5, 0.7, 0.9, 0.92]

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor(COLOR_BG)

# Plot sine waves
for f, label, col in freqs:
    y = np.sin(f * x)
    ax.plot(x, y, color=col, lw=2.0, label=label, zorder=2)

# Sample points
for xp in sample_points:
    ax.axvline(xp, color="#aaaaaa", ls="--", lw=0.8, zorder=1)
    ha = "right" if xp == 0.9 else ("left" if xp == 0.92 else "center")
    ax.text(xp, -1.35, f"x={xp}", ha=ha, va="top",
            fontsize=9, fontweight="bold", color=COLOR_TEXT)
    for f, _, col in freqs:
        yp = np.sin(f * xp)
        ax.plot(xp, yp, 'o', color=col, ms=6, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)

ax.set_xlabel("Pozycja x na boisku (znormalizowana)", fontsize=11,
              fontweight="bold", color=COLOR_TEXT)
ax.set_ylabel("Wartość sin(f·x)", fontsize=11,
              fontweight="bold", color=COLOR_TEXT)
ax.set_title("Kodowanie Fouriera dla trzech częstotliwości", fontsize=13,
             fontweight="bold", color=COLOR_TEXT, pad=10)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-1.5, 1.5)
ax.tick_params(colors=COLOR_TEXT, labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#cccccc")

leg = ax.legend(fontsize=10, loc="upper left", framealpha=0.9,
                edgecolor="#cccccc", fancybox=True,
                prop={"weight": "bold"})
leg.get_frame().set_facecolor("white")

fig.tight_layout()
fig.savefig("thesis_figures/viz_fourier_encoding.png", dpi=200,
            bbox_inches="tight", facecolor=COLOR_BG)
plt.close(fig)
print("Saved viz_fourier_encoding.png")
