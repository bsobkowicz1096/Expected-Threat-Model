"""
Re-evaluate Phase 4 & 5 models with 1000 Monte Carlo rollouts.
Loads saved MLflow model checkpoints — no retraining needed.

Why 1000 rollouts only for these:
  Phases 1-3 optimized data/capacity — relative ranking is stable even at 100 rollouts.
  Phase 4-5 compare architectural shape (dual vs single head, carry vs pass-only)
  where the effect sizes are smaller and MC noise at 100 rollouts can flip conclusions.

Usage:
    python reeval_1000_rollouts.py
"""

import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import mlflow

# We need both model classes
# Import carry_v2 model (dual/single MDN)
sys.path.insert(0, '.')
import train_xt_model_carry_v2 as carry_module
import train_xt_model as passonly_module

N_ROLLOUTS = 1000
EXPERIMENT_NAME = "xT_Phase4_5_Reeval_1000rollouts"

MODELS = [
    {
        "name": "dual_mdn_t2_ctx8_L4_v2",
        "model_path": "mlruns/696892981165273777/models/m-c36fa9de5cbe437f92e78d2806dd2b22/artifacts/data/model.pth",
        "module": "carry",
        "single_mdn": False,
        "vocab_path": "data/vocab_continuous_carry.json",
        "val_path": "data/sequences_carry2_val_natural.parquet",
        "eval_max_seq_len": 16,
        "notes": "Reeval 1000 rollouts: dual MDN carry model",
    },
    {
        "name": "single_mdn_t2_ctx8_L4_v2",
        "model_path": "mlruns/696892981165273777/models/m-00d31e86998542edb6c1ba063af290d7/artifacts/data/model.pth",
        "module": "carry",
        "single_mdn": True,
        "vocab_path": "data/vocab_continuous_carry.json",
        "val_path": "data/sequences_carry2_val_natural.parquet",
        "eval_max_seq_len": 16,
        "notes": "Reeval 1000 rollouts: single MDN carry model",
    },
    {
        "name": "passonly_ctx8_layers4",
        "model_path": "mlruns/531041205651785772/models/m-5696fea29bda41b682c381b03e056112/artifacts/data/model.pth",
        "module": "passonly",
        "single_mdn": None,
        "vocab_path": "data/vocab_continuous_passonly.json",
        "val_path": "data/sequences_passonly_ctx8_val_natural.parquet",
        "eval_max_seq_len": 10,
        "notes": "Reeval 1000 rollouts: pass-only baseline",
    },
]


def load_carry_model(model_path, single_mdn, vocab_size, device):
    # MLflow saves full model via cloudpickle — load directly
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def load_passonly_model(model_path, vocab_size, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Rollouts: {N_ROLLOUTS}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    for cfg in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {cfg['name']}")
        print(f"{'='*60}")

        # Load vocab
        with open(cfg['vocab_path'], 'r') as f:
            type_vocab = json.load(f)
        vocab_size = len(type_vocab)

        # Load model
        if cfg['module'] == 'carry':
            model = load_carry_model(cfg['model_path'], cfg['single_mdn'], vocab_size, device)
            evaluate_xT = carry_module.evaluate_xT
            DatasetClass = carry_module.ContinuousXTDataset
        else:
            model = load_passonly_model(cfg['model_path'], vocab_size, device)
            evaluate_xT = passonly_module.evaluate_xT
            DatasetClass = passonly_module.ContinuousXTDataset

        # Load val data
        df_val = pd.read_parquet(cfg['val_path'])
        eval_dataset = DatasetClass(df_val, type_vocab, max_seq_len=cfg['eval_max_seq_len'])

        print(f"Val sequences: {len(df_val):,}")
        print(f"Val goals: {df_val['goal'].sum()} ({df_val['goal'].mean()*100:.1f}%)")
        print(f"eval_max_seq_len: {cfg['eval_max_seq_len']}")

        # Evaluate
        start_time = time.time()
        print(f"Start: {pd.Timestamp.now()}")

        metrics = evaluate_xT(
            model, eval_dataset, type_vocab,
            N_ROLLOUTS, cfg['eval_max_seq_len'], device
        )

        elapsed = time.time() - start_time
        print(f"Done in {elapsed/60:.1f} min")

        print(f"\nResults:")
        print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
        print(f"  Brier Score:      {metrics['brier_score']:.4f}")
        print(f"  Mean xT (goals):  {metrics['mean_xT_goals']:.4f}")
        print(f"  Mean xT (no goals): {metrics['mean_xT_no_goals']:.4f}")
        print(f"  Separation:       {metrics['separation']:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name=cfg['name']):
            mlflow.log_param("n_rollouts", N_ROLLOUTS)
            mlflow.log_param("model_source", cfg['model_path'])
            mlflow.log_param("eval_max_seq_len", cfg['eval_max_seq_len'])
            mlflow.log_param("val_path", cfg['val_path'])
            if cfg['single_mdn'] is not None:
                mlflow.log_param("single_mdn", cfg['single_mdn'])
            mlflow.log_param("module", cfg['module'])
            mlflow.log_param("notes", cfg['notes'])

            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.log_metric("eval_time_min", elapsed / 60)

    print(f"\n{'='*60}")
    print(f"All re-evaluations complete!")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
