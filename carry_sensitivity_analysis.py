"""
Event Distance Sensitivity Analysis (Carry & Pass)

Takes 1000 real sequences from the validation set, appends a synthetic
Carry or Pass with distance ranging from 0 to 5 (StatsBomb units),
and measures how xT changes as a function of event distance.

Runs both event types separately to compare their signal thresholds.

Usage:
    python carry_sensitivity_analysis.py
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train_xt_model_carry import ContinuousXTModel, parse_mdn_params

# ============================================================================
# Config
# ============================================================================
MODEL_PATH = "best_model_carry.pt"
VAL_PATH = "data/sequences_carrynofilter_val_natural.parquet"
VOCAB_PATH = "data/vocab_continuous_carry.json"

N_SAMPLES = 1000
N_ROLLOUTS = 100
MAX_SEQ_LEN = 16
DISTANCES_SB = np.arange(0, 5.1, 0.25)  # StatsBomb units (0 to 5)
EVENT_TYPES = ['Carry', 'Pass']

SEED = 42


# ============================================================================
# xT calculation (adapted from train_xt_model_carry.py)
# ============================================================================
def calculate_xT(model, types, positions, start_mask, type_vocab,
                 n_rollouts=100, max_seq_len=16, device='cuda'):
    model.eval()
    with torch.no_grad():
        seq_types = types.unsqueeze(0).repeat(n_rollouts, 1)
        seq_positions = positions.unsqueeze(0).repeat(n_rollouts, 1, 1)
        seq_start_mask = start_mask.unsqueeze(0).repeat(n_rollouts, 1)

        active = torch.ones(n_rollouts, dtype=torch.bool, device=device)
        goals = torch.zeros(n_rollouts, dtype=torch.bool, device=device)

        for step in range(10):
            if not active.any():
                break

            type_logits, mdn_params = model(seq_types, seq_positions, seq_start_mask)

            last_logits = type_logits[:, -1, :]
            type_probs = F.softmax(last_logits, dim=-1)
            next_types = torch.multinomial(type_probs, 1).squeeze(-1)

            is_goal = (next_types == type_vocab['GOAL']) & active
            is_no_goal = (next_types == type_vocab['NO_GOAL']) & active

            goals |= is_goal
            active &= ~(is_goal | is_no_goal)

            if not active.any():
                break

            parsed = parse_mdn_params(mdn_params[:, -1:])
            weights = parsed['weights'][:, 0, :]
            k = torch.multinomial(weights, 1).squeeze(-1)

            batch_idx = torch.arange(n_rollouts, device=device)
            start_mean = parsed['start_mean'][batch_idx, 0, k]
            start_std = parsed['start_std'][batch_idx, 0, k]
            end_mean = parsed['end_mean'][batch_idx, 0, k]
            end_std = parsed['end_std'][batch_idx, 0, k]

            start_xy = (start_mean + torch.randn_like(start_mean) * start_std.unsqueeze(-1)).clamp(0, 1)
            end_xy = (end_mean + torch.randn_like(end_mean) * end_std).clamp(0, 1)

            is_shot = (next_types == type_vocab['Shot'])
            end_xy[is_shot] = 0.0

            new_pos = torch.cat([start_xy, end_xy], dim=-1).unsqueeze(1)
            new_types = next_types.unsqueeze(1)
            new_mask = torch.ones(n_rollouts, 1, dtype=torch.bool, device=device)

            seq_types = torch.cat([seq_types, new_types], dim=1)
            seq_positions = torch.cat([seq_positions, new_pos], dim=1)
            seq_start_mask = torch.cat([seq_start_mask, new_mask], dim=1)

            if seq_types.size(1) >= max_seq_len:
                break

        return goals.float().mean().item()


def build_sequence_tensors(events, type_vocab, device):
    """Convert event list to model input tensors (no causal shift, no padding)."""
    type_ids = [type_vocab[e['type']] for e in events]
    positions = []
    for e in events:
        positions.append([
            e['x'] if e['x'] is not None else 0.0,
            e['y'] if e['y'] is not None else 0.0,
            e['end_x'] if e['end_x'] is not None else 0.0,
            e['end_y'] if e['end_y'] is not None else 0.0
        ])
    start_mask = [e['x'] is not None for e in events]

    return (
        torch.tensor(type_ids, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.float32, device=device),
        torch.tensor(start_mask, dtype=torch.bool, device=device)
    )


def sb_distance_to_normalized_dx(dist_sb):
    """
    Convert StatsBomb distance to normalized x-displacement.
    SB pitch is 120 units wide, so normalized dx = dist_sb / 120.
    """
    return dist_sb / 120.0


def append_event(events, event_type, distance_sb):
    """
    Append a synthetic event (Carry or Pass) to the end of a sequence.
    The event starts at the last event's end position and moves forward by distance_sb.
    """
    # Find last non-terminal event with valid end position
    last_event = None
    for e in reversed(events):
        if e['type'] not in ('GOAL', 'NO_GOAL') and e['end_x'] is not None:
            last_event = e
            break

    if last_event is None:
        return None

    start_x = last_event['end_x']
    start_y = last_event['end_y']

    dx = sb_distance_to_normalized_dx(distance_sb)
    end_x = min(start_x + dx, 1.0)  # clamp to pitch
    end_y = start_y  # straight forward

    new_event = {
        'type': event_type,
        'x': start_x,
        'y': start_y,
        'end_x': end_x,
        'end_y': end_y
    }

    # Keep only non-terminal events + append new event
    non_terminal = [e for e in events if e['type'] not in ('GOAL', 'NO_GOAL')]
    return non_terminal + [new_event]


# ============================================================================
# Main
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load vocab and model
    with open(VOCAB_PATH, 'r') as f:
        type_vocab = json.load(f)

    num_classes = len(type_vocab)
    model = ContinuousXTModel(
        vocab_size=num_classes, d_model=512, nhead=8,
        num_layers=12, n_components=5, dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    # Load validation data and sample
    df = pd.read_parquet(VAL_PATH)
    np.random.seed(SEED)
    sample_indices = np.random.choice(len(df), size=N_SAMPLES, replace=False)
    print(f"Sampled {N_SAMPLES} sequences from {len(df)} total.")

    # Storage: {event_type: {distance: [xT values]}}
    all_results = {et: {d: [] for d in DISTANCES_SB} for et in EVENT_TYPES}
    baseline_xTs = []

    # Pre-filter valid sequences
    valid_sequences = []
    for idx in sample_indices:
        events = df.iloc[idx]['events']
        input_events = [e for e in events if e['type'] not in ('GOAL', 'NO_GOAL')]

        if len(input_events) >= MAX_SEQ_LEN - 1:
            continue
        if not any(e['end_x'] is not None for e in input_events):
            continue

        valid_sequences.append(events)

    print(f"Valid sequences after filtering: {len(valid_sequences)}")

    for seq_num, events in enumerate(valid_sequences):
        input_events = [e for e in events if e['type'] not in ('GOAL', 'NO_GOAL')]

        # Baseline xT (no appended event)
        types, positions, mask = build_sequence_tensors(input_events, type_vocab, device)
        baseline_xt = calculate_xT(model, types, positions, mask, type_vocab,
                                   N_ROLLOUTS, MAX_SEQ_LEN, device)
        baseline_xTs.append(baseline_xt)

        # For each event type and distance
        for event_type in EVENT_TYPES:
            for dist in DISTANCES_SB:
                new_events = append_event(events, event_type, dist)
                if new_events is None or len(new_events) >= MAX_SEQ_LEN:
                    all_results[event_type][dist].append(np.nan)
                    continue

                types_e, positions_e, mask_e = build_sequence_tensors(
                    new_events, type_vocab, device
                )
                xt = calculate_xT(model, types_e, positions_e, mask_e, type_vocab,
                                  N_ROLLOUTS, MAX_SEQ_LEN, device)
                all_results[event_type][dist].append(xt)

        if (seq_num + 1) % 50 == 0:
            print(f"Processed {seq_num + 1}/{len(valid_sequences)}")

    baseline_mean = np.mean(baseline_xTs)

    # Compute stats per event type
    summary = {}
    for event_type in EVENT_TYPES:
        mean_xTs = []
        std_xTs = []
        for dist in DISTANCES_SB:
            vals = [v for v in all_results[event_type][dist] if not np.isnan(v)]
            mean_xTs.append(np.mean(vals) if vals else np.nan)
            std_xTs.append(np.std(vals) if vals else np.nan)
        summary[event_type] = {
            'mean': np.array(mean_xTs),
            'std': np.array(std_xTs)
        }

    # Print results
    print(f"\nBaseline mean xT (no appended event): {baseline_mean:.4f}")
    for event_type in EVENT_TYPES:
        print(f"\n{'='*60}")
        print(f"  {event_type} Sensitivity")
        print(f"{'='*60}")
        print(f"{'Distance (SB)':>15} | {'Mean xT':>10} | {'Delta vs baseline':>18} | {'Std':>8}")
        print("-" * 60)
        for i, dist in enumerate(DISTANCES_SB):
            m = summary[event_type]['mean'][i]
            delta = m - baseline_mean
            s = summary[event_type]['std'][i]
            print(f"{dist:>15.2f} | {m:>10.4f} | {delta:>+18.4f} | {s:>8.4f}")

    # Save results to CSV
    rows = []
    for event_type in EVENT_TYPES:
        for i, dist in enumerate(DISTANCES_SB):
            rows.append({
                'event_type': event_type,
                'distance_sb': dist,
                'mean_xT': summary[event_type]['mean'][i],
                'std_xT': summary[event_type]['std'][i],
                'delta_vs_baseline': summary[event_type]['mean'][i] - baseline_mean
            })
    results_df = pd.DataFrame(rows)
    results_df.to_csv('sensitivity_results.csv', index=False)
    print("\nResults saved to sensitivity_results.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'Carry': 'blue', 'Pass': 'orange'}

    # Plot 1: Mean xT vs distance (both types)
    ax1 = axes[0]
    for event_type in EVENT_TYPES:
        m = summary[event_type]['mean']
        s = summary[event_type]['std']
        ax1.plot(DISTANCES_SB, m, '-o', markersize=4, color=colors[event_type],
                 label=event_type)
        ax1.fill_between(DISTANCES_SB, m - s, m + s, alpha=0.15, color=colors[event_type])
    ax1.axhline(y=baseline_mean, color='r', linestyle='--',
                label=f'Baseline: {baseline_mean:.4f}')
    ax1.set_xlabel('Distance (StatsBomb units)')
    ax1.set_ylabel('Mean xT')
    ax1.set_title('xT vs Event Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Delta xT vs distance (both types)
    ax2 = axes[1]
    for event_type in EVENT_TYPES:
        delta = summary[event_type]['mean'] - baseline_mean
        ax2.plot(DISTANCES_SB, delta, '-o', markersize=4, color=colors[event_type],
                 label=event_type)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Distance (StatsBomb units)')
    ax2.set_ylabel('Delta xT (vs baseline)')
    ax2.set_title('Event Impact on xT')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Carry vs Pass delta difference
    ax3 = axes[2]
    carry_delta = summary['Carry']['mean'] - baseline_mean
    pass_delta = summary['Pass']['mean'] - baseline_mean
    ax3.plot(DISTANCES_SB, carry_delta - pass_delta, '-o', markersize=4, color='green')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Distance (StatsBomb units)')
    ax3.set_ylabel('Carry delta - Pass delta')
    ax3.set_title('Carry vs Pass: Differential Impact')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensitivity_plot.png', dpi=150, bbox_inches='tight')
    print("Plot saved to sensitivity_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
