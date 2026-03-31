"""
Expected Threat (xT) Model Training Script — Carry Challenger v2 (Dual MDN)
Based on train_xt_model_carry.py with SEPARATE MDN heads for Pass and Carry.

Key change vs v1:
  - mdn_head_pass: 5 components (passes have diverse spatial distribution)
  - mdn_head_carry: 3 components (carries are shorter, more constrained)
  - During training: target type routes loss to the correct head
  - During rollout: sampled type selects which head to draw location from
"""

import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss
import mlflow


# ============================================================================
# Dataset (unchanged from v1)
# ============================================================================

class ContinuousXTDataset(Dataset):
    def __init__(self, df, type_vocab, max_seq_len=16):
        self.df = df.reset_index(drop=True)
        self.type_vocab = type_vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        events = row['events']

        # Causal shift
        input_events = events[:-1]
        target_events = events[1:]
        seq_len = len(input_events)

        # Type IDs
        input_type_ids = [self.type_vocab[e['type']] for e in input_events]
        target_type_ids = [self.type_vocab[e['type']] for e in target_events]

        # Positions [start_x, start_y, end_x, end_y]
        def _build_positions(events_list):
            positions = []
            for e in events_list:
                pos = [
                    e['x'] if e['x'] is not None else 0.0,
                    e['y'] if e['y'] is not None else 0.0,
                    e['end_x'] if e['end_x'] is not None else 0.0,
                    e['end_y'] if e['end_y'] is not None else 0.0
                ]
                positions.append(pos)
            return positions

        input_positions = _build_positions(input_events)
        target_positions = _build_positions(target_events)

        # Masks
        def _build_masks(events_list):
            start_masks = [e['x'] is not None for e in events_list]
            end_masks = [e['end_x'] is not None for e in events_list]
            return start_masks, end_masks

        input_start_mask, input_end_mask = _build_masks(input_events)
        target_start_mask, target_end_mask = _build_masks(target_events)

        # Padding
        pad_len = self.max_seq_len - seq_len
        input_type_ids += [self.type_vocab['<pad>']] * pad_len
        target_type_ids += [-100] * pad_len
        input_positions += [[0.0, 0.0, 0.0, 0.0]] * pad_len
        target_positions += [[0.0, 0.0, 0.0, 0.0]] * pad_len
        input_start_mask += [False] * pad_len
        input_end_mask += [False] * pad_len
        target_start_mask += [False] * pad_len
        target_end_mask += [False] * pad_len

        return {
            'input_types': torch.tensor(input_type_ids, dtype=torch.long),
            'input_positions': torch.tensor(input_positions, dtype=torch.float32),
            'input_start_mask': torch.tensor(input_start_mask, dtype=torch.bool),
            'input_end_mask': torch.tensor(input_end_mask, dtype=torch.bool),
            'target_types': torch.tensor(target_type_ids, dtype=torch.long),
            'target_positions': torch.tensor(target_positions, dtype=torch.float32),
            'target_start_mask': torch.tensor(target_start_mask, dtype=torch.bool),
            'target_end_mask': torch.tensor(target_end_mask, dtype=torch.bool)
        }


# ============================================================================
# Model Components
# ============================================================================

class FourierPositionEncoder(nn.Module):
    def __init__(self, freqs=[1,2,4,8,16,32,64,128], d_model=512):
        super().__init__()
        self.freqs = torch.tensor(freqs, dtype=torch.float32)
        self.proj = nn.Linear(64, d_model)

    def forward(self, pos):
        B, T, _ = pos.shape
        freqs = self.freqs.to(pos.device)
        pos_expanded = pos.unsqueeze(-1)
        angles = pos_expanded * freqs
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        fourier = torch.stack([sin_features, cos_features], dim=-1)
        fourier = fourier.reshape(B, T, -1)
        return self.proj(fourier)


class ContinuousXTModel(nn.Module):
    def __init__(self, vocab_size=6, d_model=512, nhead=8, num_layers=12,
                 n_components_pass=5, n_components_carry=3, dropout=0.1,
                 single_mdn=False):
        super().__init__()
        self.n_components_pass = n_components_pass
        self.n_components_carry = n_components_carry
        self.single_mdn = single_mdn

        # Embeddings
        self.type_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoder = FourierPositionEncoder(d_model=d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        self.type_head = nn.Linear(d_model, vocab_size)
        self.mdn_head_pass = nn.Linear(d_model, n_components_pass * 8)
        if not single_mdn:
            self.mdn_head_carry = nn.Linear(d_model, n_components_carry * 8)

    def forward(self, types, positions, start_mask):
        type_emb = self.type_embedding(types)
        pos_emb = self.position_encoder(positions)
        pos_emb = pos_emb * start_mask.unsqueeze(-1).float()
        combined = type_emb + pos_emb

        T = types.size(1)
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(types.device)
        hidden = self.transformer(combined, mask=causal_mask)

        type_logits = self.type_head(hidden)

        B = types.size(0)
        mdn_pass = self.mdn_head_pass(hidden).view(B, T, self.n_components_pass, 8)

        if self.single_mdn:
            # Single head for all spatial tokens — return pass head as carry too
            mdn_carry = mdn_pass[:, :, :self.n_components_carry, :]
        else:
            mdn_carry = self.mdn_head_carry(hidden).view(B, T, self.n_components_carry, 8)

        return type_logits, mdn_pass, mdn_carry


# ============================================================================
# Loss Functions
# ============================================================================

def parse_mdn_params(mdn_params):
    weights = F.softmax(mdn_params[..., 0], dim=-1)
    start_mean_x = torch.sigmoid(mdn_params[..., 1])
    start_mean_y = torch.sigmoid(mdn_params[..., 2])
    start_std = torch.exp(mdn_params[..., 3]).clamp(0.005, 0.1)
    end_mean_x = torch.sigmoid(mdn_params[..., 4])
    end_mean_y = torch.sigmoid(mdn_params[..., 5])
    end_std_x = torch.exp(mdn_params[..., 6]).clamp(0.01, 0.5)
    end_std_y = torch.exp(mdn_params[..., 7]).clamp(0.01, 0.5)

    return {
        'weights': weights,
        'start_mean': torch.stack([start_mean_x, start_mean_y], dim=-1),
        'start_std': start_std,
        'end_mean': torch.stack([end_mean_x, end_mean_y], dim=-1),
        'end_std': torch.stack([end_std_x, end_std_y], dim=-1)
    }


def type_loss(type_logits, target_types, class_weights, num_classes):
    return F.cross_entropy(
        type_logits.reshape(-1, num_classes),
        target_types.reshape(-1),
        weight=class_weights,
        ignore_index=-100
    )


def gaussian_nll(target, mean, std):
    variance = std ** 2
    return 0.5 * (torch.log(2 * torch.pi * variance) + ((target - mean) ** 2) / variance)


def _mdn_nll(mdn_params, target_positions):
    """Compute per-token MDN NLL (no masking)."""
    parsed = parse_mdn_params(mdn_params)
    n_components = mdn_params.shape[2]

    target_start = target_positions[..., :2]
    target_end = target_positions[..., 2:]

    component_nll = []
    for k in range(n_components):
        start_nll = gaussian_nll(
            target_start.unsqueeze(2),
            parsed['start_mean'][:, :, k:k+1, :],
            parsed['start_std'][:, :, k:k+1].unsqueeze(-1)
        ).sum(dim=-1)

        end_nll = gaussian_nll(
            target_end.unsqueeze(2),
            parsed['end_mean'][:, :, k:k+1, :],
            parsed['end_std'][:, :, k:k+1, :]
        ).sum(dim=-1)

        component_nll.append(start_nll + end_nll)

    component_nll = torch.cat(component_nll, dim=-1)
    log_weights = torch.log(parsed['weights'] + 1e-8)
    mixture_nll = -torch.logsumexp(log_weights - component_nll, dim=-1)

    return mixture_nll


def dual_mdn_loss(mdn_pass, mdn_carry, target_positions, target_start_mask,
                  target_types, pass_id, carry_id):
    """
    Compute MDN loss routing each token to the correct head.
    Pass tokens -> mdn_pass, Carry tokens -> mdn_carry.
    """
    # Compute NLL from both heads
    nll_pass = _mdn_nll(mdn_pass, target_positions)    # (B, T)
    nll_carry = _mdn_nll(mdn_carry, target_positions)  # (B, T)

    # Build masks for each type
    is_pass = (target_types == pass_id).float()
    is_carry = (target_types == carry_id).float()
    spatial_mask = target_start_mask.float()

    # Route: use pass NLL for pass tokens, carry NLL for carry tokens
    # Shots also use pass head for start location (they have start_mask=True)
    routed_nll = nll_pass * (spatial_mask - is_carry) + nll_carry * is_carry

    return routed_nll.sum() / (spatial_mask.sum() + 1e-8)


def combined_loss(model, batch, class_weights, num_classes, loss_ratio, pass_id, carry_id):
    """
    Combined loss with dual MDN heads.
    """
    type_logits, mdn_pass, mdn_carry = model(
        batch['input_types'],
        batch['input_positions'],
        batch['input_start_mask']
    )

    t_loss = type_loss(type_logits, batch['target_types'], class_weights, num_classes)
    m_loss = dual_mdn_loss(
        mdn_pass, mdn_carry,
        batch['target_positions'], batch['target_start_mask'],
        batch['target_types'], pass_id, carry_id
    )

    total_loss = loss_ratio * t_loss + m_loss
    return total_loss, t_loss, m_loss


# ============================================================================
# xT Calculation
# ============================================================================

def calculate_xT_montecarlo(model, start_sequence, type_vocab, n_rollouts=100,
                            max_steps=10, max_seq_len=16, device='cuda'):
    model.eval()

    with torch.no_grad():
        seq_types = start_sequence['types'].unsqueeze(0).repeat(n_rollouts, 1)
        seq_positions = start_sequence['positions'].unsqueeze(0).repeat(n_rollouts, 1, 1)
        seq_start_mask = start_sequence['start_mask'].unsqueeze(0).repeat(n_rollouts, 1)

        active = torch.ones(n_rollouts, dtype=torch.bool, device=device)
        goals = torch.zeros(n_rollouts, dtype=torch.bool, device=device)

        for step in range(max_steps):
            if not active.any():
                break

            type_logits, mdn_pass, mdn_carry = model(seq_types, seq_positions, seq_start_mask)

            last_logits = type_logits[:, -1, :]
            type_probs = F.softmax(last_logits, dim=-1)
            next_types = torch.multinomial(type_probs, 1).squeeze(-1)

            is_goal = (next_types == type_vocab['GOAL']) & active
            is_no_goal = (next_types == type_vocab['NO_GOAL']) & active

            goals |= is_goal
            active &= ~(is_goal | is_no_goal)

            if not active.any():
                break

            # Route to correct MDN head based on sampled type
            is_carry = (next_types == type_vocab['Carry'])
            is_pass = (next_types == type_vocab['Pass'])
            is_shot = (next_types == type_vocab['Shot'])

            # Sample from pass head (default)
            parsed_pass = parse_mdn_params(mdn_pass[:, -1:])
            weights_pass = parsed_pass['weights'][:, 0, :]
            k_pass = torch.multinomial(weights_pass, 1).squeeze(-1)

            batch_idx = torch.arange(n_rollouts, device=device)
            start_mean = parsed_pass['start_mean'][batch_idx, 0, k_pass]
            start_std = parsed_pass['start_std'][batch_idx, 0, k_pass]
            end_mean = parsed_pass['end_mean'][batch_idx, 0, k_pass]
            end_std = parsed_pass['end_std'][batch_idx, 0, k_pass]

            # Override carry tokens with carry head
            if is_carry.any():
                parsed_carry = parse_mdn_params(mdn_carry[:, -1:])
                weights_carry = parsed_carry['weights'][:, 0, :]
                k_carry = torch.multinomial(weights_carry, 1).squeeze(-1)

                carry_start_mean = parsed_carry['start_mean'][batch_idx, 0, k_carry]
                carry_start_std = parsed_carry['start_std'][batch_idx, 0, k_carry]
                carry_end_mean = parsed_carry['end_mean'][batch_idx, 0, k_carry]
                carry_end_std = parsed_carry['end_std'][batch_idx, 0, k_carry]

                start_mean[is_carry] = carry_start_mean[is_carry]
                start_std[is_carry] = carry_start_std[is_carry]
                end_mean[is_carry] = carry_end_mean[is_carry]
                end_std[is_carry] = carry_end_std[is_carry]

            start_xy = (start_mean + torch.randn_like(start_mean) * start_std.unsqueeze(-1)).clamp(0, 1)
            end_xy = (end_mean + torch.randn_like(end_mean) * end_std).clamp(0, 1)

            # Shot has no end location
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


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, class_weights, num_classes, loss_ratio,
                pass_id, carry_id, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _, _ = combined_loss(model, batch, class_weights, num_classes,
                                   loss_ratio, pass_id, carry_id)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, class_weights, num_classes, loss_ratio,
             pass_id, carry_id, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _, _ = combined_loss(model, batch, class_weights, num_classes,
                                       loss_ratio, pass_id, carry_id)
            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_xT(model, dataset, type_vocab, n_rollouts, max_seq_len, device):
    val_xTs = []
    val_labels = []

    for i in range(len(dataset)):
        sample = dataset[i]

        real_len = (sample['input_types'] != type_vocab['<pad>']).sum().item()

        if real_len <= 2:
            start_len = real_len
        else:
            start_len = 3

        start_seq = {
            'types': sample['input_types'][:start_len].to(device),
            'positions': sample['input_positions'][:start_len].to(device),
            'start_mask': sample['input_start_mask'][:start_len].to(device)
        }

        xT = calculate_xT_montecarlo(
            model, start_seq, type_vocab, n_rollouts,
            max_seq_len=max_seq_len, device=device
        )
        val_xTs.append(xT)

        label = (sample['target_types'] == type_vocab['GOAL']).any().item()
        val_labels.append(label)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(dataset)}")

    val_xTs = np.array(val_xTs)
    val_labels = np.array(val_labels)

    roc_auc = roc_auc_score(val_labels, val_xTs)
    brier = brier_score_loss(val_labels, val_xTs)
    mean_xT_goals = val_xTs[val_labels==1].mean()
    mean_xT_no_goals = val_xTs[val_labels==0].mean()
    separation = mean_xT_goals - mean_xT_no_goals

    return {
        'roc_auc': roc_auc,
        'brier_score': brier,
        'mean_xT_goals': mean_xT_goals,
        'mean_xT_no_goals': mean_xT_no_goals,
        'separation': separation
    }


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    df_train = pd.read_parquet(args.train_path)
    df_val = pd.read_parquet(args.val_path)

    with open(args.vocab_path, 'r') as f:
        type_vocab = json.load(f)

    num_classes = len(type_vocab)
    pass_id = type_vocab['Pass']
    carry_id = type_vocab['Carry']

    print(f"\nVocab size: {num_classes}")
    print(f"Vocab: {type_vocab}")
    print(f"Pass ID: {pass_id}, Carry ID: {carry_id}")
    print(f"\nTrain sequences: {len(df_train):,}")
    print(f"Train goals: {df_train['goal'].sum()} ({df_train['goal'].mean()*100:.1f}%)")
    print(f"\nVal sequences: {len(df_val):,}")
    print(f"Val goals: {df_val['goal'].sum()} ({df_val['goal'].mean()*100:.1f}%)")

    # Datasets
    max_seq_len = args.max_seq_len
    eval_max_seq_len = args.eval_max_seq_len or max_seq_len
    train_dataset = ContinuousXTDataset(df_train, type_vocab, max_seq_len=max_seq_len)
    val_dataset = ContinuousXTDataset(df_val, type_vocab, max_seq_len=max_seq_len)

    # Separate eval dataset (full sequences) for xT evaluation
    if args.eval_val_path:
        df_eval_val = pd.read_parquet(args.eval_val_path)
        eval_val_dataset = ContinuousXTDataset(df_eval_val, type_vocab, max_seq_len=eval_max_seq_len)
        print(f"\nEval val sequences (full): {len(df_eval_val):,}")
        print(f"Eval val goals: {df_eval_val['goal'].sum()} ({df_eval_val['goal'].mean()*100:.1f}%)")
        print(f"Eval max_seq_len: {eval_max_seq_len}")
    else:
        eval_val_dataset = val_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )

    # Weight config for logging
    weight_config = {
        "Pass": 1.0,
        "Shot": args.weight_shot,
        "GOAL": args.weight_goal,
        "NO_GOAL": args.weight_no_goal,
        "<pad>": 1.0,
        "Carry": args.weight_carry
    }

    # Build class weights tensor in vocab order
    id_to_type = {v: k for k, v in type_vocab.items()}
    class_weights = torch.tensor([
        weight_config.get(id_to_type[i], 1.0) for i in range(num_classes)
    ]).to(device)

    print(f"\nClass weights: {dict(zip([id_to_type[i] for i in range(num_classes)], class_weights.tolist()))}")

    # Model
    print("Building model...")
    print(f"MDN components: pass={args.n_components_pass}, carry={args.n_components_carry}")
    model = ContinuousXTModel(
        vocab_size=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        n_components_pass=args.n_components_pass,
        n_components_carry=args.n_components_carry,
        dropout=args.dropout,
        single_mdn=args.single_mdn
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # MLflow
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model", "Transformer_SingleMDN" if args.single_mdn else "Transformer_DualMDN")
        mlflow.log_param("variant", "single_mdn_ablation" if args.single_mdn else "carry_challenger_v2")
        mlflow.log_param("single_mdn", args.single_mdn)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("d_model", args.d_model)
        mlflow.log_param("num_layers", args.num_layers)
        mlflow.log_param("n_components_pass", args.n_components_pass)
        mlflow.log_param("n_components_carry", args.n_components_carry)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("max_seq_len", max_seq_len)
        mlflow.log_param("eval_max_seq_len", eval_max_seq_len)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("loss_ratio", args.loss_ratio)
        mlflow.log_param("loss_weights", str(weight_config))
        if args.notes:
            mlflow.log_param("notes", args.notes)

        # Training loop
        loss_ratio = args.loss_ratio
        print(f"\nTraining (loss_ratio={loss_ratio})...")
        print(f"Start time: {pd.Timestamp.now()}")
        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, class_weights, num_classes,
                loss_ratio, pass_id, carry_id, device
            )
            val_loss = evaluate(
                model, val_loader, class_weights, num_classes,
                loss_ratio, pass_id, carry_id, device
            )

            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model_carry_v2.pt')

        # Load best model
        model.load_state_dict(torch.load('best_model_carry_v2.pt'))

        print(f"\nTraining completed: {pd.Timestamp.now()}")

        # Final evaluation with xT (always on full sequences)
        print("\nEvaluating xT...")
        print(f"Evaluation start: {pd.Timestamp.now()}")
        metrics = evaluate_xT(
            model, eval_val_dataset, type_vocab, args.n_rollouts, eval_max_seq_len, device
        )

        print(f"Evaluation completed: {pd.Timestamp.now()}")

        print(f"\nFinal Metrics:")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Mean xT (goals): {metrics['mean_xT_goals']:.4f}")
        print(f"Mean xT (no goals): {metrics['mean_xT_no_goals']:.4f}")
        print(f"Separation: {metrics['separation']:.4f}")

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.pytorch.log_model(model, "model")
        print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xT model with Carry events — Dual MDN v2")

    # Data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--eval_val_path", type=str, default=None,
                        help="Separate val path for xT evaluation (full sequences). If not set, uses --val_path")
    parser.add_argument("--eval_max_seq_len", type=int, default=None,
                        help="max_seq_len for evaluation dataset. If not set, uses --max_seq_len")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--n_components_pass", type=int, default=5)
    parser.add_argument("--n_components_carry", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--single_mdn", action="store_true",
                        help="Use single MDN head for all spatial tokens (ablation)")
    parser.add_argument("--max_seq_len", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Loss weights
    parser.add_argument("--weight_shot", type=float, default=3.0)
    parser.add_argument("--weight_goal", type=float, default=15.0)
    parser.add_argument("--weight_no_goal", type=float, default=1.5)
    parser.add_argument("--weight_carry", type=float, default=1.0)
    parser.add_argument("--loss_ratio", type=float, default=1.0,
                        help="Weight for type_loss relative to mdn_loss")

    # Evaluation
    parser.add_argument("--n_rollouts", type=int, default=100)

    # MLflow
    parser.add_argument("--experiment_name", type=str, default="xT_with_Carry_test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)

    args = parser.parse_args()
    main(args)
