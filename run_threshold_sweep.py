"""
Phase 1: Threshold sweep with dual MDN v2, uniform weights.
Generates data for each threshold, then trains with identical hyperparams.

Usage:
    python run_threshold_sweep.py
"""

import subprocess
import sys

THRESHOLDS = [1, 2, 3]
EXPERIMENT_NAME = "xT_Carry_Phase1_Threshold_Sweep"
VOCAB_PATH = "data/vocab_continuous_carry.json"

PARAMS = {
    "d_model": 512,
    "num_layers": 12,
    "nhead": 8,
    "n_components_pass": 5,
    "n_components_carry": 3,
    "dropout": 0.1,
    "max_seq_len": 16,
    "epochs": 15,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "weight_shot": 1.0,
    "weight_goal": 1.0,
    "weight_no_goal": 1.0,
    "weight_carry": 1.0,
    "loss_ratio": 1.0,
    "n_rollouts": 100,
}


for t in THRESHOLDS:
    # Step 1: Generate data
    print(f"\n{'='*60}")
    print(f"Generating data for threshold {t}")
    print(f"{'='*60}")

    subprocess.run([
        sys.executable, "generate_carry_data.py",
        "--carry_threshold", str(t),
    ], check=True)

    # Step 2: Train
    train_path = f"data/sequences_carry{t}_train_balanced.parquet"
    val_path = f"data/sequences_carry{t}_val_natural.parquet"
    run_name = f"dual_mdn_t{t}_uniform"

    cmd = [
        sys.executable, "train_xt_model_carry_v2.py",
        "--train_path", train_path,
        "--val_path", val_path,
        "--vocab_path", VOCAB_PATH,
        "--experiment_name", EXPERIMENT_NAME,
        "--run_name", run_name,
        "--notes", f"Phase 1: dual MDN v2, threshold >= {t}, uniform weights, 15 epochs",
    ]

    for key, value in PARAMS.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"{'='*60}")

    subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print("Phase 1 threshold sweep complete!")
print(f"MLflow experiment: {EXPERIMENT_NAME}")
print(f"{'='*60}")
