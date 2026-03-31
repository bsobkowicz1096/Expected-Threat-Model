"""
Phase 4: Head ablation — dual MDN vs single MDN.
Tests whether a separate carry MDN head improves over a shared head.

Fixed: t=2, ctx=8, 4 layers, uniform weights, 15 epochs.
Evaluation on full val set (ctx=14, max_seq_len=16).

Usage:
    python run_head_ablation.py
"""

import subprocess
import sys

THRESHOLD = 2
TRAIN_CTX = 8
TRAIN_MAX_SEQ_LEN = TRAIN_CTX + 2  # 10
EVAL_MAX_SEQ_LEN = 16  # full sequences (ctx=14 + terminal + pad)
NUM_LAYERS = 4

EXPERIMENT_NAME = "xT_Carry_Phase4_Head_Ablation"
VOCAB_PATH = "data/vocab_continuous_carry.json"

# Data paths (ctx=8 data from Phase 2)
TRAIN_PATH = "data/sequences_carry2_ctx8_train_balanced.parquet"
VAL_PATH = "data/sequences_carry2_ctx8_val_natural.parquet"
EVAL_VAL_PATH = "data/sequences_carry2_val_natural.parquet"

PARAMS = {
    "d_model": 512,
    "nhead": 8,
    "num_layers": NUM_LAYERS,
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

VARIANTS = [
    {"name": "dual_mdn_t2_ctx8_L4_v2", "single_mdn": False,
     "notes": "Phase 4 rerun: dual MDN (baseline), t>=2, ctx=8, 4 layers"},
    {"name": "single_mdn_t2_ctx8_L4_v2", "single_mdn": True,
     "notes": "Phase 4 rerun: single MDN (ablation), t>=2, ctx=8, 4 layers"},
]

for variant in VARIANTS:
    cmd = [
        sys.executable, "train_xt_model_carry_v2.py",
        "--train_path", TRAIN_PATH,
        "--val_path", VAL_PATH,
        "--vocab_path", VOCAB_PATH,
        "--eval_val_path", EVAL_VAL_PATH,
        "--eval_max_seq_len", str(EVAL_MAX_SEQ_LEN),
        "--experiment_name", EXPERIMENT_NAME,
        "--run_name", variant["name"],
        "--notes", variant["notes"],
    ]

    if variant["single_mdn"]:
        cmd.append("--single_mdn")

    for key, value in PARAMS.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Training: {variant['name']}")
    print(f"{'='*60}")

    subprocess.run(cmd, check=True)

print(f"\n{'='*60}")
print("Phase 4 head ablation complete!")
print(f"MLflow experiment: {EXPERIMENT_NAME}")
print(f"{'='*60}")
