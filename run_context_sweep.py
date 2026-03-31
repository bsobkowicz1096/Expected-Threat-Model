"""
Phase 2: Context length sweep with dual MDN v2, uniform weights.
Tests different context lengths with the best threshold from Phase 1.

Training uses truncated data per context length.
Evaluation always uses the SAME full-length val set (ctx=14) for fair comparison.

Usage:
    python run_context_sweep.py --threshold 2
"""

import argparse
import subprocess
import sys
import os

# Context lengths = max events per possession (before terminal token)
# Model max_seq_len = context_length + 2 (terminal + padding)
CONTEXT_LENGTHS = [2, 5, 8, 10, 14, 18, 23]

# Eval always uses full sequences (ctx=14 -> max_seq_len=16)
EVAL_CONTEXT = 14
EVAL_MAX_SEQ_LEN = EVAL_CONTEXT + 2  # 16

EXPERIMENT_NAME = "xT_Carry_Phase2_Context_Sweep"
VOCAB_PATH = "data/vocab_continuous_carry.json"

PARAMS = {
    "d_model": 512,
    "num_layers": 12,
    "nhead": 8,
    "n_components_pass": 5,
    "n_components_carry": 3,
    "dropout": 0.1,
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


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Context length sweep")
    parser.add_argument("--threshold", type=int, required=True,
                        help="Best threshold from Phase 1 (e.g. 2 or 3)")
    args = parser.parse_args()

    t = args.threshold

    # Full-length eval val path (from Phase 1 or ctx=14 generation)
    eval_val_path = f"data/sequences_carry{t}_val_natural.parquet"
    if not os.path.exists(eval_val_path):
        print(f"Generating full-length eval data (ctx={EVAL_CONTEXT})...")
        subprocess.run([
            sys.executable, "generate_carry_data.py",
            "--carry_threshold", str(t),
            "--max_seq_len", str(EVAL_CONTEXT),
        ], check=True)

    print(f"\nEval val set (shared): {eval_val_path}")
    print(f"Eval max_seq_len: {EVAL_MAX_SEQ_LEN}")

    for ctx_len in CONTEXT_LENGTHS:
        model_max_seq_len = ctx_len + 2  # events + terminal + padding
        ctx_suffix = f"_ctx{ctx_len}" if ctx_len != 14 else ""
        suffix = f"{t}{ctx_suffix}"

        # Step 1: Generate training data with this context length
        if ctx_len != EVAL_CONTEXT:
            print(f"\n{'='*60}")
            print(f"Generating training data: threshold={t}, context_length={ctx_len}")
            print(f"{'='*60}")

            subprocess.run([
                sys.executable, "generate_carry_data.py",
                "--carry_threshold", str(t),
                "--max_seq_len", str(ctx_len),
            ], check=True)

        # Step 2: Train (with truncated data) + Evaluate (on full val set)
        train_path = f"data/sequences_carry{suffix}_train_balanced.parquet"
        val_path = f"data/sequences_carry{suffix}_val_natural.parquet"
        run_name = f"dual_mdn_t{t}_ctx{ctx_len}"

        cmd = [
            sys.executable, "train_xt_model_carry_v2.py",
            "--train_path", train_path,
            "--val_path", val_path,
            "--vocab_path", VOCAB_PATH,
            "--max_seq_len", str(model_max_seq_len),
            "--eval_val_path", eval_val_path,
            "--eval_max_seq_len", str(EVAL_MAX_SEQ_LEN),
            "--experiment_name", EXPERIMENT_NAME,
            "--run_name", run_name,
            "--notes", f"Phase 2: dual MDN v2, threshold >= {t}, train_ctx={ctx_len}, eval on full sequences (ctx={EVAL_CONTEXT})",
        ]

        for key, value in PARAMS.items():
            cmd.extend([f"--{key}", str(value)])

        print(f"\n{'='*60}")
        print(f"Training: {run_name} (train max_seq_len={model_max_seq_len}, eval max_seq_len={EVAL_MAX_SEQ_LEN})")
        print(f"{'='*60}")

        subprocess.run(cmd, check=True)

    print(f"\n{'='*60}")
    print("Phase 2 context length sweep complete!")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
