"""
Run carry experiments with thresholds 1, 2, 3 and uniform weights (all 1.0).
Data for threshold 3 already exists; generates data for 1 and 2 first.

Usage:
    conda run -n pyspark311 python run_carry_noweights.py
"""

import subprocess
import sys
import os

THRESHOLDS = [1, 2, 3]
EXPERIMENT_NAME = "xT_with_Carry_test"
VOCAB_PATH = "data/vocab_continuous_carry.json"

HYPERPARAMS = {
    "d_model": 512,
    "num_layers": 12,
    "nhead": 8,
    "n_components": 5,
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
    "n_rollouts": 100,
}

# Step 1: Generate data for thresholds that don't have data yet
for t in THRESHOLDS:
    train_path = f"data/sequences_carry{t}_train_balanced.parquet"
    if not os.path.exists(train_path):
        print(f"\n{'='*60}")
        print(f"Generating data for threshold {t}")
        print(f"{'='*60}")
        subprocess.run([
            sys.executable, "generate_carry_data.py",
            "--carry_threshold", str(t)
        ], check=True)
    else:
        print(f"Data for threshold {t} already exists, skipping generation.")

# Step 2: Train all three with uniform weights
for t in THRESHOLDS:
    train_path = f"data/sequences_carry{t}_train_balanced.parquet"
    val_path = f"data/sequences_carry{t}_val_natural.parquet"
    run_name = f"carry_t{t}_noweights"

    cmd = [
        sys.executable, "train_xt_model_carry.py",
        "--train_path", train_path,
        "--val_path", val_path,
        "--vocab_path", VOCAB_PATH,
        "--experiment_name", EXPERIMENT_NAME,
        "--run_name", run_name,
        "--notes", f"Carry threshold >= {t}, uniform weights (all 1.0), 15 epochs",
    ]

    for key, value in HYPERPARAMS.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Data: {train_path}")
    print(f"{'='*60}")

    subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print("All carry no-weights experiments complete!")
print(f"Check MLflow experiment: {EXPERIMENT_NAME}")
print(f"{'='*60}")
