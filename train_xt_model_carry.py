"""
Expected Threat (xT) Model Training Script — Carry Challenger
Based on train_xt_model.py with support for 6 event classes (Pass, Shot, GOAL, NO_GOAL, <pad>, Carry).
Key changes vs baseline:
  - Dynamic num_classes from vocab (not hardcoded 5)
  - max_seq_len = 16 (up from 14)
  - --weight_carry argument
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
# Dataset
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
                 n_components=5, dropout=0.1):
        super().__init__()
        self.n_components = n_components

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
        self.mdn_head = nn.Linear(d_model, n_components * 8)

    def forward(self, types, positions, start_mask):
        type_emb = self.type_embedding(types)
        pos_emb = self.position_encoder(positions)
        pos_emb = pos_emb * start_mask.unsqueeze(-1).float()
        combined = type_emb + pos_emb

        T = types.size(1)
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(types.device)
        hidden = self.transformer(combined, mask=causal_mask)

        type_logits = self.type_head(hidden)
        mdn_params = self.mdn_head(hidden).view(types.size(0), T, self.n_components, 8)

        return type_logits, mdn_params


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


def mdn_loss(mdn_params, target_positions, target_start_mask):
    parsed = parse_mdn_params(mdn_params)
    B, T, n_components = mdn_params.shape[:3]

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

    mask = target_start_mask.float()
    return (mixture_nll * mask).sum() / (mask.sum() + 1e-8)


def combined_loss(model, batch, class_weights, num_classes, loss_ratio=1.0):
    """
    Combined loss with configurable ratio between type_loss and mdn_loss.

    loss_ratio=0.3 -> type has 30% weight of MDN
    loss_ratio=1.0 -> equal weights (baseline)
    loss_ratio=2.0 -> type has 2x weight of MDN
    """
    type_logits, mdn_params = model(
        batch['input_types'],
        batch['input_positions'],
        batch['input_start_mask']
    )

    t_loss = type_loss(type_logits, batch['target_types'], class_weights, num_classes)
    m_loss = mdn_loss(mdn_params, batch['target_positions'], batch['target_start_mask'])

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

            # Shot has no end location
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


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, class_weights, num_classes, loss_ratio, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _, _ = combined_loss(model, batch, class_weights, num_classes, loss_ratio)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, class_weights, num_classes, loss_ratio, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _, _ = combined_loss(model, batch, class_weights, num_classes, loss_ratio)
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
    print(f"\nVocab size: {num_classes}")
    print(f"Vocab: {type_vocab}")
    print(f"\nTrain sequences: {len(df_train):,}")
    print(f"Train goals: {df_train['goal'].sum()} ({df_train['goal'].mean()*100:.1f}%)")
    print(f"\nVal sequences: {len(df_val):,}")
    print(f"Val goals: {df_val['goal'].sum()} ({df_val['goal'].mean()*100:.1f}%)")

    # Datasets
    max_seq_len = args.max_seq_len
    train_dataset = ContinuousXTDataset(df_train, type_vocab, max_seq_len=max_seq_len)
    val_dataset = ContinuousXTDataset(df_val, type_vocab, max_seq_len=max_seq_len)

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
    model = ContinuousXTModel(
        vocab_size=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        n_components=args.n_components,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # MLflow
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model", "Transformer_MDN")
        mlflow.log_param("variant", "carry_challenger")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("d_model", args.d_model)
        mlflow.log_param("num_layers", args.num_layers)
        mlflow.log_param("n_components", args.n_components)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("max_seq_len", max_seq_len)
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
                model, train_loader, optimizer, class_weights, num_classes, loss_ratio, device
            )
            val_loss = evaluate(model, val_loader, class_weights, num_classes, loss_ratio, device)

            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model_carry.pt')

        # Load best model
        model.load_state_dict(torch.load('best_model_carry.pt'))

        print(f"\nTraining completed: {pd.Timestamp.now()}")

        # Final evaluation with xT
        print("\nEvaluating xT...")
        print(f"Evaluation start: {pd.Timestamp.now()}")
        metrics = evaluate_xT(
            model, val_dataset, type_vocab, args.n_rollouts, max_seq_len, device
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
    parser = argparse.ArgumentParser(description="Train xT model with Carry events")

    # Data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)

    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
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
