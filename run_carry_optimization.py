"""
Unified carry model optimization script.
Stage 1: Grid search over class loss weights (weight_goal, weight_shot, weight_no_goal, weight_carry)
Stage 2: Grid search over lr and loss_ratio (using best weights from Stage 1)

Usage:
    python run_carry_optimization.py --stage 1 --data_suffix 3
    python run_carry_optimization.py --stage 2 --data_suffix 3 \
        --weight_goal 15 --weight_shot 3 --weight_no_goal 1.5 --weight_carry 1.0
"""

import argparse
import subprocess
import sys
from itertools import product


VOCAB_PATH = "data/vocab_continuous_carry.json"
EXPERIMENT_NAME = "xT_Carry_Optimization"

# Fixed architecture params (same as baseline)
FIXED = {
    "d_model": 512,
    "num_layers": 12,
    "nhead": 8,
    "n_components": 5,
    "dropout": 0.1,
    "max_seq_len": 16,
    "batch_size": 32,
    "weight_decay": 0.01,
    "n_rollouts": 100,
}


def build_stage1_grid():
    """Grid search over class loss weights."""
    return {
        "weight_goal": [10, 15, 20, 25],
        "weight_shot": [1, 3, 5],
        "weight_no_goal": [1.0, 1.5, 2.0],
        "weight_carry": [1.0, 2.0, 3.0],
    }


def build_stage2_grid():
    """Grid search over learning rate and loss_ratio."""
    return {
        "lr": [5e-5, 1e-4, 2e-4],
        "loss_ratio": [0.3, 0.5, 1.0, 2.0, 3.0],
    }


def run_experiment(config, data_suffix, stage):
    train_path = f"data/sequences_carry{data_suffix}_train_balanced.parquet"
    val_path = f"data/sequences_carry{data_suffix}_val_natural.parquet"

    cmd = [
        sys.executable, "train_xt_model_carry.py",
        "--train_path", train_path,
        "--val_path", val_path,
        "--vocab_path", VOCAB_PATH,
        "--experiment_name", EXPERIMENT_NAME,
        "--run_name", config["run_name"],
        "--notes", f"Stage {stage}, carry data threshold {data_suffix}",
    ]

    # Add all params
    for key in FIXED:
        cmd.extend([f"--{key}", str(config.get(key, FIXED[key]))])

    for key in ["weight_goal", "weight_shot", "weight_no_goal", "weight_carry",
                "lr", "loss_ratio", "epochs"]:
        if key in config:
            cmd.extend([f"--{key}", str(config[key])])

    return cmd


def run_stage1(data_suffix):
    grid = build_stage1_grid()
    keys = list(grid.keys())
    values = list(grid.values())

    configs = []
    for combo in product(*values):
        config = {k: v for k, v in zip(keys, combo)}
        config["lr"] = 1e-4
        config["loss_ratio"] = 1.0
        config["epochs"] = 10

        g, s, n, c = config["weight_goal"], config["weight_shot"], \
                      config["weight_no_goal"], config["weight_carry"]
        config["run_name"] = f"s1_g{g}_s{s}_n{n}_c{c}"
        configs.append(config)

    total = len(configs)
    print(f"Stage 1: {total} experiments")
    print(f"Grid: {grid}")
    print(f"Data: carry threshold {data_suffix}")
    print("=" * 60)

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total}] Running: {config['run_name']}")
        cmd = run_experiment(config, data_suffix, stage=1)
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed {i}/{total}")
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\nStage 1 completed! {total} experiments in {EXPERIMENT_NAME}")


def run_stage2(data_suffix, weight_goal, weight_shot, weight_no_goal, weight_carry):
    grid = build_stage2_grid()

    configs = []
    for lr, loss_ratio in product(grid["lr"], grid["loss_ratio"]):
        lr_str = f"{lr:.0e}".replace("e-0", "e-")
        ratio_str = f"{loss_ratio:.1f}".replace(".", "_")
        config = {
            "lr": lr,
            "loss_ratio": loss_ratio,
            "weight_goal": weight_goal,
            "weight_shot": weight_shot,
            "weight_no_goal": weight_no_goal,
            "weight_carry": weight_carry,
            "epochs": 12,
            "run_name": f"s2_lr{lr_str}_r{ratio_str}",
        }
        configs.append(config)

    total = len(configs)
    print(f"Stage 2: {total} experiments")
    print(f"Grid: {grid}")
    print(f"Fixed weights: goal={weight_goal}, shot={weight_shot}, "
          f"no_goal={weight_no_goal}, carry={weight_carry}")
    print(f"Data: carry threshold {data_suffix}")
    print("=" * 60)

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{total}] Running: {config['run_name']}")
        cmd = run_experiment(config, data_suffix, stage=2)
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed {i}/{total}")
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\nStage 2 completed! {total} experiments in {EXPERIMENT_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carry model optimization")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--data_suffix", type=str, required=True,
                        help="Carry threshold suffix (3, 5, or 8)")

    # Stage 2 requires best weights from Stage 1
    parser.add_argument("--weight_goal", type=float, default=15.0)
    parser.add_argument("--weight_shot", type=float, default=3.0)
    parser.add_argument("--weight_no_goal", type=float, default=1.5)
    parser.add_argument("--weight_carry", type=float, default=1.0)

    args = parser.parse_args()

    if args.stage == 1:
        run_stage1(args.data_suffix)
    else:
        run_stage2(
            args.data_suffix,
            args.weight_goal, args.weight_shot,
            args.weight_no_goal, args.weight_carry
        )
