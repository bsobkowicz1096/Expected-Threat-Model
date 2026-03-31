"""
Phase 3: Architecture sweep — number of Transformer layers.
Tests whether 12 layers is overkill for short context (ctx=8).

Fixed: t=2, ctx=8, uniform weights, 15 epochs.
Evaluation on full val set (ctx=14, max_seq_len=16).

Usage:
    python run_architecture_sweep.py
"""

import subprocess
import sys

NUM_LAYERS_LIST = [4, 8, 16]

THRESHOLD = 2
TRAIN_CTX = 8
TRAIN_MAX_SEQ_LEN = TRAIN_CTX + 2  # 10
EVAL_MAX_SEQ_LEN = 16  # full sequences (ctx=14 + terminal + pad)

EXPERIMENT_NAME = "xT_Carry_Phase3_Architecture_Sweep"
VOCAB_PATH = "data/vocab_continuous_carry.json"

# Data paths (ctx=8 data already generated from Phase 2)
TRAIN_PATH = "data/sequences_carry2_ctx8_train_balanced.parquet"
VAL_PATH = "data/sequences_carry2_ctx8_val_natural.parquet"
EVAL_VAL_PATH = "data/sequences_carry2_val_natural.parquet"

PARAMS = {
    "d_model": 512,
    "nhead": 8,
    "n_components_pass": 5,
    "n_components_carry": 3,
    "dropout": 0.1,
    "max_seq_len": TRAIN_MAX_SEQ_LEN,
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

for n_layers in NUM_LAYERS_LIST:
    run_name = f"dual_mdn_t2_ctx8_layers{n_layers}"

    cmd = [
        sys.executable, "train_xt_model_carry_v2.py",
        "--train_path", TRAIN_PATH,
        "--val_path", VAL_PATH,
        "--vocab_path", VOCAB_PATH,
        "--eval_val_path", EVAL_VAL_PATH,
        "--eval_max_seq_len", str(EVAL_MAX_SEQ_LEN),
        "--num_layers", str(n_layers),
        "--experiment_name", EXPERIMENT_NAME,
        "--run_name", run_name,
        "--notes", f"Phase 3: {n_layers} layers, t>=2, ctx=8, uniform weights",
    ]

    for key, value in PARAMS.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"{'='*60}")

    subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print("Phase 3 architecture sweep complete!")
print(f"MLflow experiment: {EXPERIMENT_NAME}")
print(f"{'='*60}")
