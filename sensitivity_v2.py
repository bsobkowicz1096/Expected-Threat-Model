"""
Sensitivity Analysis v2
Model: nofilter, weights=1.0
Base: 3 tokens, then append Carry or Pass at varying distances (0-5 SB units)
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_xt_model_carry import ContinuousXTModel, parse_mdn_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

with open('data/vocab_continuous_carry.json', 'r') as f:
    type_vocab = json.load(f)

model = ContinuousXTModel(vocab_size=6, d_model=512, nhead=8, num_layers=12, n_components=5, dropout=0.1).to(device)
model.load_state_dict(torch.load('best_model_nofilter_w1.pt', map_location=device, weights_only=True))
model.eval()
print('Model: nofilter, weights=1.0')

df = pd.read_parquet('data/sequences_carrynofilter_val_natural.parquet')
np.random.seed(42)
indices = np.random.choice(len(df), size=1000, replace=False)

DISTANCES_SB = np.arange(0, 5.1, 0.25)
EVENT_TYPES = ['Carry', 'Pass']


def calc_xT(model, types, positions, start_mask, type_vocab, n_rollouts=100, max_seq_len=16, device='cuda'):
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
            seq_types = torch.cat([seq_types, next_types.unsqueeze(1)], dim=1)
            seq_positions = torch.cat([seq_positions, new_pos], dim=1)
            seq_start_mask = torch.cat([seq_start_mask, torch.ones(n_rollouts, 1, dtype=torch.bool, device=device)], dim=1)
            if seq_types.size(1) >= max_seq_len:
                break
        return goals.float().mean().item()


def to_tensors(events, type_vocab, device):
    type_ids = [type_vocab[e['type']] for e in events]
    positions = [[e['x'] or 0, e['y'] or 0, e['end_x'] or 0, e['end_y'] or 0] for e in events]
    mask = [e['x'] is not None for e in events]
    return (
        torch.tensor(type_ids, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device)
    )


# Collect valid sequences (min 3 non-terminal events with valid end positions)
valid = []
for idx in indices:
    events = df.iloc[idx]['events']
    non_term = [e for e in events if e['type'] not in ('GOAL', 'NO_GOAL')]
    if len(non_term) < 3:
        continue
    last = None
    for e in reversed(non_term[:3]):
        if e['end_x'] is not None:
            last = e
            break
    if last is None:
        continue
    valid.append(non_term[:3])

print(f'Valid sequences: {len(valid)}')

# Compute
all_results = {et: {d: [] for d in DISTANCES_SB} for et in EVENT_TYPES}
baseline_xTs = []

for seq_num, base_events in enumerate(valid):
    # Baseline: 3 tokens
    t, p, m = to_tensors(base_events, type_vocab, device)
    base_xt = calc_xT(model, t, p, m, type_vocab, 100, 16, device)
    baseline_xTs.append(base_xt)

    # End position of last base event
    last = None
    for e in reversed(base_events):
        if e['end_x'] is not None:
            last = e
            break
    start_x = last['end_x']
    start_y = last['end_y']

    for event_type in EVENT_TYPES:
        for dist in DISTANCES_SB:
            dx = dist / 120.0
            end_x = min(start_x + dx, 1.0)
            new_event = {
                'type': event_type,
                'x': start_x, 'y': start_y,
                'end_x': end_x, 'end_y': start_y
            }
            new_events = base_events + [new_event]
            t2, p2, m2 = to_tensors(new_events, type_vocab, device)
            xt = calc_xT(model, t2, p2, m2, type_vocab, 100, 16, device)
            all_results[event_type][dist].append(xt)

    if (seq_num + 1) % 50 == 0:
        print(f'Processed {seq_num + 1}/{len(valid)}')

baseline_mean = np.mean(baseline_xTs)

# Stats
summary = {}
for et in EVENT_TYPES:
    means = []
    stds = []
    for dist in DISTANCES_SB:
        vals = all_results[et][dist]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    summary[et] = {'mean': np.array(means), 'std': np.array(stds)}

# Print
print(f'\nBaseline mean xT (3 tokens): {baseline_mean:.4f}')
for et in EVENT_TYPES:
    print(f'\n  {et}:')
    print(f'  {"Dist":>6} | {"Mean xT":>8} | {"Delta":>8}')
    print('  ' + '-' * 30)
    for i, dist in enumerate(DISTANCES_SB):
        m_val = summary[et]['mean'][i]
        print(f'  {dist:>6.2f} | {m_val:>8.4f} | {m_val - baseline_mean:>+8.4f}')

# Save CSV
rows = []
for et in EVENT_TYPES:
    for i, dist in enumerate(DISTANCES_SB):
        rows.append({
            'event_type': et,
            'distance_sb': dist,
            'mean_xT': summary[et]['mean'][i],
            'std_xT': summary[et]['std'][i],
            'delta_vs_baseline': summary[et]['mean'][i] - baseline_mean
        })
pd.DataFrame(rows).to_csv('sensitivity_results_v2.csv', index=False)
print('\nResults saved to sensitivity_results_v2.csv')

# Plot
colors = {'Carry': 'blue', 'Pass': 'orange'}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1 = axes[0]
for et in EVENT_TYPES:
    m_arr = summary[et]['mean']
    s_arr = summary[et]['std']
    ax1.plot(DISTANCES_SB, m_arr, '-o', markersize=4, color=colors[et], label=et)
    ax1.fill_between(DISTANCES_SB, m_arr - s_arr, m_arr + s_arr, alpha=0.15, color=colors[et])
ax1.axhline(y=baseline_mean, color='r', linestyle='--', label=f'Baseline: {baseline_mean:.4f}')
ax1.set_xlabel('Distance (StatsBomb units)')
ax1.set_ylabel('Mean xT')
ax1.set_title('xT vs Event Distance')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for et in EVENT_TYPES:
    delta = summary[et]['mean'] - baseline_mean
    ax2.plot(DISTANCES_SB, delta, '-o', markersize=4, color=colors[et], label=et)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Distance (StatsBomb units)')
ax2.set_ylabel('Delta xT (vs baseline)')
ax2.set_title('Event Impact on xT')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
carry_d = summary['Carry']['mean'] - baseline_mean
pass_d = summary['Pass']['mean'] - baseline_mean
ax3.plot(DISTANCES_SB, carry_d - pass_d, '-o', markersize=4, color='green')
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Distance (StatsBomb units)')
ax3.set_ylabel('Carry delta - Pass delta')
ax3.set_title('Carry vs Pass: Differential Impact')
ax3.grid(True, alpha=0.3)

plt.suptitle('Sensitivity Analysis v2 — Model: nofilter, weights=1.0 | Base: 3 tokens', fontsize=13)
plt.tight_layout()
plt.savefig('sensitivity_plot_v2.png', dpi=150, bbox_inches='tight')
print('Plot saved to sensitivity_plot_v2.png')
