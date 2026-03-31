"""
EDA module for Expected Threat Model project.
Functions for exploratory data analysis on StatsBomb event data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mplsoccer import Pitch


# ---------------------------------------------------------------------------
# Helpers / style
# ---------------------------------------------------------------------------

INFERNO = cm.get_cmap("inferno")
# Pick 3 evenly-spaced colours from inferno for bar charts
COLOR_1 = INFERNO(0.3)   # dark warm
COLOR_2 = INFERNO(0.55)  # mid orange
COLOR_3 = INFERNO(0.8)   # bright yellow
ACCENT  = INFERNO(0.95)  # highlight / threshold lines

RELEVANT_COLUMNS = [
    "type", "location", "pass_end_location", "carry_end_location",
    "shot_outcome", "possession", "team", "possession_team", "match_id",
]

MOVEMENT_TYPES = ["Pass", "Carry", "Shot"]


def load_raw(path: str = "data/all_events_combined_2015_2016.parquet",
             columns: list[str] | None = None) -> pd.DataFrame:
    """Load raw StatsBomb combined events parquet."""
    cols = columns or RELEVANT_COLUMNS
    return pd.read_parquet(path, columns=cols)


def _extract_xy(arr):
    """Extract (x, y) from a numpy array or return (NaN, NaN)."""
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return np.nan, np.nan
    return float(arr[0]), float(arr[1])


# ---------------------------------------------------------------------------
# 1. Null / completeness analysis
# ---------------------------------------------------------------------------

def null_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with null counts and percentages for relevant columns."""
    total = len(df)
    records = []
    for col in df.columns:
        if col == "location":
            n_null = df[col].apply(lambda v: v is None or (isinstance(v, float) and np.isnan(v))).sum()
        elif col in ("pass_end_location", "carry_end_location", "shot_outcome"):
            # These are expected to be null for non-matching event types
            n_null = df[col].isna().sum()
        else:
            n_null = df[col].isna().sum()
        records.append({
            "column": col,
            "non_null": total - n_null,
            "null": n_null,
            "null_pct": round(100 * n_null / total, 2),
        })
    return pd.DataFrame(records)


def null_analysis_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    For position columns, show null rate only for the event types where
    they *should* be populated (Pass->pass_end_location, Carry->carry_end_location, etc.).
    """
    checks = [
        ("location", None, "All events"),
        ("pass_end_location", "Pass", "Pass events"),
        ("carry_end_location", "Carry", "Carry events"),
        ("shot_outcome", "Shot", "Shot events"),
    ]
    records = []
    for col, etype, label in checks:
        subset = df if etype is None else df[df["type"] == etype]
        total = len(subset)
        if col == "location":
            n_null = subset[col].apply(
                lambda v: v is None or (isinstance(v, float) and np.isnan(v))
            ).sum()
        else:
            n_null = subset[col].isna().sum()
        records.append({
            "column": col,
            "scope": label,
            "total": total,
            "null": n_null,
            "null_pct": round(100 * n_null / total, 4),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Event type distribution
# ---------------------------------------------------------------------------

def plot_movement_event_distribution(df: pd.DataFrame,
                                      ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart of movement event types only (Pass, Carry, Shot)."""
    subset = df[df["type"].isin(MOVEMENT_TYPES)]
    counts = subset["type"].value_counts().reindex(MOVEMENT_TYPES)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    colors = {"Pass": COLOR_2, "Carry": COLOR_1, "Shot": COLOR_3}
    bars = ax.bar(counts.index, counts.values,
                  color=[colors[t] for t in counts.index], edgecolor="white")
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() + 150_000)
    ax.set_title("Movement Events Used by Model")
    ax.ticklabel_format(axis="y", style="plain")
    return ax


# ---------------------------------------------------------------------------
# 3. Possession chain lengths
# ---------------------------------------------------------------------------

def compute_possession_lengths(df: pd.DataFrame, movement_only: bool = False) -> pd.Series:
    """
    Compute possession chain lengths.
    If movement_only=True, count only Pass/Carry/Shot events per possession.
    Returns a Series indexed by (match_id, possession).
    """
    subset = df if not movement_only else df[df["type"].isin(MOVEMENT_TYPES)]
    return subset.groupby(["match_id", "possession"]).size()


def plot_possession_length_distribution(lengths: pd.Series,
                                         title: str = "Possession Chain Lengths",
                                         ax: plt.Axes | None = None,
                                         max_display: int = 50) -> plt.Axes:
    """Histogram of possession chain lengths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    clipped = lengths.clip(upper=max_display)
    ax.hist(clipped, bins=range(1, max_display + 2), color=COLOR_2,
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of events")
    ax.set_ylabel("Number of possessions")
    ax.set_title(f"{title} (n={len(lengths):,})")
    mean_val = lengths.mean()
    median_val = lengths.median()
    ax.axvline(mean_val, color=COLOR_1, linestyle="--", linewidth=1.5)
    ax.axvline(median_val, color=COLOR_1, linestyle="-", linewidth=1.5)
    ylim = ax.get_ylim()[1]
    ax.text(mean_val - 0.3, ylim * 0.95, f"mean={mean_val:.1f}",
            fontsize=9, color=COLOR_1, fontweight="bold", rotation=90, va="top", ha="right")
    ax.text(median_val - 0.3, ylim * 0.95, f"median={median_val:.0f}",
            fontsize=9, color=COLOR_1, fontweight="bold", rotation=90, va="top", ha="right")
    if lengths.max() > max_display:
        ax.axvline(max_display, color=COLOR_1, linestyle="--", alpha=0.5)
        ax.text(max_display - 0.5, ylim * 0.6,
                f"clipped at {max_display}\n(max={lengths.max()})",
                ha="right", fontsize=8, color=COLOR_1)
    return ax


def compute_goal_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute goal rate per possession. Returns DataFrame with goal flag per possession."""
    # A possession has a goal if any Shot event in it has shot_outcome == 'Goal'
    shots = df[df["type"] == "Shot"].copy()
    goals_per_poss = shots.groupby(["match_id", "possession"])["shot_outcome"].apply(
        lambda x: (x == "Goal").any().astype(int)
    ).reset_index(name="goal")
    # Total possessions
    all_poss = df.groupby(["match_id", "possession"]).size().reset_index(name="n_events")
    merged = all_poss.merge(goals_per_poss, on=["match_id", "possession"], how="left")
    merged["goal"] = merged["goal"].fillna(0).astype(int)
    return merged


def _possession_funnel_stats(df: pd.DataFrame) -> dict:
    """Compute possession → shot → goal counts."""
    n_poss = df.groupby(["match_id", "possession"]).ngroups
    shots = df[df["type"] == "Shot"]
    poss_with_shot = shots[["match_id", "possession"]].drop_duplicates()
    n_with_shot = len(poss_with_shot)
    n_with_goal = len(
        shots[shots["shot_outcome"] == "Goal"][["match_id", "possession"]].drop_duplicates()
    )
    return {"possessions": n_poss, "with_shot": n_with_shot, "with_goal": n_with_goal}


def plot_shot_goal_funnel(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Funnel chart: all possessions → with shot → with goal."""
    s = _possession_funnel_stats(df)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["All possessions", "With shot", "With goal"]
    values = [s["possessions"], s["with_shot"], s["with_goal"]]
    colors = [COLOR_2, COLOR_3, INFERNO(0.95)]

    y_positions = [2, 1, 0]
    max_width = 1.0
    widths = [max_width * v / values[0] for v in values]

    for y, w, c, label, val in zip(y_positions, widths, colors, labels, values):
        ax.barh(y, w, height=0.6, color=c, edgecolor="white", left=-w / 2)
        pct = 100 * val / values[0]
        ax.text(0, y, f"{label}\n{val:,} ({pct:.1f}%)",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if w > 0.15 else "black")

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.5, 2.8)
    ax.set_title("Possession Outcome Funnel")
    ax.axis("off")
    return ax


def plot_shot_goal_waffle(df: pd.DataFrame, ax: plt.Axes | None = None,
                          cols: int = 10, rows: int = 10) -> plt.Axes:
    """Waffle chart: shot and goal proportions among possessions.
    cols x rows grid; fractional cells are partially filled."""
    s = _possession_funnel_stats(df)
    total_cells = cols * rows
    pct_goal_exact = 100 * s["with_goal"] / s["possessions"]
    pct_shot_exact = 100 * (s["with_shot"] - s["with_goal"]) / s["possessions"]
    pct_no_shot = 100 - pct_goal_exact - pct_shot_exact

    # Scale percentages to cell counts (rounded)
    goal_cells = round(pct_goal_exact * total_cells / 100)
    shot_cells = round(pct_shot_exact * total_cells / 100)
    goal_cells = max(goal_cells, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    from matplotlib.patches import Rectangle

    def cell_xy(idx):
        row, col = divmod(idx, cols)
        return col, row

    for i in range(total_cells):
        cx, cy = cell_xy(i)
        if i < goal_cells:
            color = COLOR_3
        elif i < goal_cells + shot_cells:
            color = COLOR_2
        else:
            color = COLOR_1
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1,
                                facecolor=color, edgecolor="white", linewidth=1))

    # Grid lines
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1, zorder=3)
    for i in range(cols + 1):
        ax.axvline(i - 0.5, color="white", linewidth=1, zorder=3)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    cell_pct = 100 / total_cells
    ax.set_title(f"1 square = {cell_pct:g}% of possessions", fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_1, label=f"No shot ({pct_no_shot:.1f}%)"),
        Patch(facecolor=COLOR_2, label=f"Shot, no goal ({pct_shot_exact:.1f}%)"),
        Patch(facecolor=COLOR_3, label=f"Goal ({pct_goal_exact:.1f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9)
    return ax


def plot_shot_goal_donut(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Double donut: outer = shot/no-shot, inner = goal/no-goal among shots."""
    s = _possession_funnel_stats(df)
    no_shot = s["possessions"] - s["with_shot"]
    shot_no_goal = s["with_shot"] - s["with_goal"]
    goal = s["with_goal"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Outer ring: shot vs no shot
    outer_vals = [s["with_shot"], no_shot]
    outer_colors = [COLOR_2, INFERNO(0.15)]
    outer_labels = [
        f"Shot\n{100 * s['with_shot'] / s['possessions']:.1f}%",
        f"No shot\n{100 * no_shot / s['possessions']:.1f}%",
    ]
    wedges1, _ = ax.pie(outer_vals, radius=1.0, colors=outer_colors,
                        wedgeprops=dict(width=0.3, edgecolor="white"),
                        startangle=90, counterclock=False)

    # Inner ring: goal vs no goal (among shots)
    inner_vals = [goal, shot_no_goal]
    inner_colors = [INFERNO(0.95), COLOR_3]
    wedges2, _ = ax.pie(inner_vals, radius=0.7, colors=inner_colors,
                        wedgeprops=dict(width=0.3, edgecolor="white"),
                        startangle=90, counterclock=False)

    ax.set_title("Possession Outcomes", fontsize=12, pad=15)

    # Legend
    from matplotlib.patches import Patch
    pct_shot = 100 * s["with_shot"] / s["possessions"]
    pct_no_shot = 100 * no_shot / s["possessions"]
    pct_goal_of_shots = 100 * goal / s["with_shot"]
    pct_no_goal_of_shots = 100 * shot_no_goal / s["with_shot"]
    legend_elements = [
        Patch(facecolor=COLOR_2, label=f"Shot ({pct_shot:.1f}% of possessions)"),
        Patch(facecolor=INFERNO(0.15), label=f"No shot ({pct_no_shot:.1f}%)"),
        Patch(facecolor=INFERNO(0.95), label=f"Goal ({pct_goal_of_shots:.1f}% of shots)"),
        Patch(facecolor=COLOR_3, label=f"No goal ({pct_no_goal_of_shots:.1f}% of shots)"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=8,
              ncol=2, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)
    return ax


# ---------------------------------------------------------------------------
# 4. Pass and carry distances
# ---------------------------------------------------------------------------

def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Euclidean distances for passes and carries.
    Returns a DataFrame with columns: type, distance.
    """
    records = []
    for etype, end_col in [("Pass", "pass_end_location"), ("Carry", "carry_end_location")]:
        subset = df[(df["type"] == etype) & df[end_col].notna()].copy()
        locs = np.stack(subset["location"].values)
        ends = np.stack(subset[end_col].values)
        dists = np.sqrt((ends[:, 0] - locs[:, 0])**2 + (ends[:, 1] - locs[:, 1])**2)
        records.append(pd.DataFrame({"type": etype, "distance": dists}))
    return pd.concat(records, ignore_index=True)


def plot_distance_distribution(distances: pd.DataFrame,
                                event_type: str = "Pass",
                                thresholds: list[float] | None = None,
                                clip: float | None = None,
                                ax: plt.Axes | None = None) -> plt.Axes:
    """Histogram of distances (Y axis in %) for a given event type."""
    subset = distances[distances["type"] == event_type]
    data = subset["distance"]
    if clip is not None:
        data = data[data <= clip]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    color = COLOR_2 if event_type == "Pass" else COLOR_1
    weights = np.ones(len(data)) / len(subset) * 100  # % of all (incl. clipped)
    ax.hist(data, bins=80, color=color, edgecolor="white", alpha=0.85, weights=weights)
    ax.set_xlabel("Distance (StatsBomb units)")
    ax.set_ylabel("% of events")
    title = f"{event_type} Distance Distribution (n={len(subset):,})"
    if clip is not None:
        title += f" [clipped at {clip}]"
    ax.set_title(title)
    ax.set_xlim(0, None)
    if thresholds:
        for t in thresholds:
            ax.axvline(t, color=ACCENT, linestyle="--", alpha=0.7)
    # Mean and median lines
    mean_val = subset["distance"].mean()
    median_val = subset["distance"].median()
    annot_color = COLOR_1 if event_type == "Pass" else COLOR_2
    ax.axvline(mean_val, color=annot_color, linestyle="--", linewidth=1.5)
    ax.axvline(median_val, color=annot_color, linestyle="-", linewidth=1.5)
    ylim = ax.get_ylim()[1]
    ax.text(mean_val - 0.3, ylim * 0.95, f"mean={mean_val:.1f}",
            fontsize=9, color=annot_color, fontweight="bold", rotation=90, va="top", ha="right")
    ax.text(median_val - 0.3, ylim * 0.95, f"median={median_val:.1f}",
            fontsize=9, color=annot_color, fontweight="bold", rotation=90, va="top", ha="right")
    return ax


def carry_threshold_summary(distances: pd.DataFrame,
                             thresholds: list[float] = [1, 2, 3, 5, 8]) -> pd.DataFrame:
    """Summary table: how many carries survive each distance threshold."""
    carries = distances[distances["type"] == "Carry"]
    total = len(carries)
    records = []
    for t in thresholds:
        n_above = (carries["distance"] >= t).sum()
        records.append({
            "threshold": t,
            "carries_remaining": n_above,
            "pct_remaining": round(100 * n_above / total, 1),
            "carries_removed": total - n_above,
            "pct_removed": round(100 * (total - n_above) / total, 1),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Heatmaps
# ---------------------------------------------------------------------------

def _pitch_heatmap(x: np.ndarray, y: np.ndarray, title: str,
                   ax: plt.Axes | None = None, bins: int = 25,
                   cmap: str = "inferno") -> plt.Axes:
    """Draw a heatmap on a soccer pitch using mplsoccer."""
    pitch = Pitch(pitch_type="statsbomb", line_color="white", line_zorder=2)
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
    else:
        pitch.draw(ax=ax)
    # Use pitch.bin_statistic for proper binning
    bin_stat = pitch.bin_statistic(x, y, statistic="count",
                                    bins=(bins, bins))
    pitch.heatmap(bin_stat, ax=ax, cmap=cmap, edgecolors="none")
    ax.set_title(title, fontsize=12, pad=10)
    return ax


def plot_shot_rate_heatmap(df: pd.DataFrame, bins: int = 12,
                           ax: plt.Axes | None = None) -> plt.Axes:
    """
    Heatmap: % of possessions passing through each zone that end with a shot.
    For each bin, counts unique possessions whose events touch that bin,
    then the fraction of those that contain a shot.
    """
    movement = df[df["type"].isin(MOVEMENT_TYPES)].copy()
    valid = movement[movement["location"].apply(
        lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
    )].copy()
    locs = np.stack(valid["location"].values)
    valid["x"] = locs[:, 0]
    valid["y"] = locs[:, 1]

    # Identify possessions that contain a shot
    poss_with_shot = (
        valid[valid["type"] == "Shot"][["match_id", "possession"]]
        .drop_duplicates()
    )
    poss_with_shot["has_shot"] = 1
    valid = valid.merge(poss_with_shot, on=["match_id", "possession"], how="left")
    valid["has_shot"] = valid["has_shot"].fillna(0)

    # Bin edges
    x_edges = np.linspace(0, 120, bins + 1)
    y_edges = np.linspace(0, 80, bins + 1)
    valid["x_bin"] = np.clip(np.digitize(valid["x"], x_edges) - 1, 0, bins - 1)
    valid["y_bin"] = np.clip(np.digitize(valid["y"], y_edges) - 1, 0, bins - 1)

    # Per bin: count unique possessions and unique possessions with shot
    grouped = (
        valid.groupby(["x_bin", "y_bin", "match_id", "possession"])["has_shot"]
        .max()
        .reset_index()
    )
    bin_stats = grouped.groupby(["x_bin", "y_bin"]).agg(
        total=("has_shot", "count"),
        with_shot=("has_shot", "sum"),
    ).reset_index()
    bin_stats["rate"] = bin_stats["with_shot"] / bin_stats["total"] * 100

    # Build grid
    grid = np.full((bins, bins), np.nan)
    for _, row in bin_stats.iterrows():
        grid[int(row["y_bin"]), int(row["x_bin"])] = row["rate"]

    pitch = Pitch(pitch_type="statsbomb", line_color="white", line_zorder=2)
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
    else:
        pitch.draw(ax=ax)

    # Use pitch bin_statistic structure for heatmap
    bin_stat = pitch.bin_statistic(valid["x"].values, valid["y"].values,
                                    statistic="count", bins=(bins, bins))
    bin_stat["statistic"] = grid
    pitch.heatmap(bin_stat, ax=ax, cmap="inferno", edgecolors="none")
    # Label with adaptive color: black on bright cells, white on dark
    cx = bin_stat["cx"]
    cy = bin_stat["cy"]
    stats = bin_stat["statistic"]
    vmax = np.nanmax(stats)
    for i in range(stats.shape[0]):
        for j in range(stats.shape[1]):
            val = stats[i, j]
            if np.isnan(val):
                continue
            color = "black" if val > vmax * 0.55 else "white"
            ax.text(cx[i, j], cy[i, j], f"{val:.1f}%",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color=color)
    ax.set_title("% of possessions passing through zone that end with a shot",
                 fontsize=12, pad=10)
    return ax


def plot_shot_rate_by_start_heatmap(df: pd.DataFrame, bins: int = 12,
                                     ax: plt.Axes | None = None) -> plt.Axes:
    """
    Heatmap: % of possessions starting in each zone that end with a shot.
    Uses the first movement event of each possession as the start location.
    """
    movement = df[df["type"].isin(MOVEMENT_TYPES)].copy()
    valid = movement[movement["location"].apply(
        lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
    )].copy()

    # First event per possession = start location
    first_events = valid.groupby(["match_id", "possession"]).first().reset_index()
    locs = np.stack(first_events["location"].values)
    first_events["x"] = locs[:, 0]
    first_events["y"] = locs[:, 1]

    # Identify possessions with a shot
    poss_with_shot = (
        valid[valid["type"] == "Shot"][["match_id", "possession"]]
        .drop_duplicates()
    )
    poss_with_shot["has_shot"] = 1
    first_events = first_events.merge(poss_with_shot, on=["match_id", "possession"], how="left")
    first_events["has_shot"] = first_events["has_shot"].fillna(0)

    # Bin
    x_edges = np.linspace(0, 120, bins + 1)
    y_edges = np.linspace(0, 80, bins + 1)
    first_events["x_bin"] = np.clip(np.digitize(first_events["x"], x_edges) - 1, 0, bins - 1)
    first_events["y_bin"] = np.clip(np.digitize(first_events["y"], y_edges) - 1, 0, bins - 1)

    bin_stats = first_events.groupby(["x_bin", "y_bin"]).agg(
        total=("has_shot", "count"),
        with_shot=("has_shot", "sum"),
    ).reset_index()
    bin_stats["rate"] = bin_stats["with_shot"] / bin_stats["total"] * 100

    grid = np.full((bins, bins), np.nan)
    for _, row in bin_stats.iterrows():
        grid[int(row["y_bin"]), int(row["x_bin"])] = row["rate"]

    pitch = Pitch(pitch_type="statsbomb", line_color="white", line_zorder=2)
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
    else:
        pitch.draw(ax=ax)

    bin_stat = pitch.bin_statistic(first_events["x"].values, first_events["y"].values,
                                    statistic="count", bins=(bins, bins))
    bin_stat["statistic"] = grid
    pitch.heatmap(bin_stat, ax=ax, cmap="inferno", edgecolors="none")
    # Label with adaptive color: black on bright cells, white on dark
    cx = bin_stat["cx"]
    cy = bin_stat["cy"]
    stats = bin_stat["statistic"]
    vmax = np.nanmax(stats)
    for i in range(stats.shape[0]):
        for j in range(stats.shape[1]):
            val = stats[i, j]
            if np.isnan(val):
                continue
            color = "black" if val > vmax * 0.55 else "white"
            ax.text(cx[i, j], cy[i, j], f"{val:.1f}%",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color=color)
    ax.set_title("% of possessions starting in zone that end with a shot",
                 fontsize=12, pad=10)
    return ax
