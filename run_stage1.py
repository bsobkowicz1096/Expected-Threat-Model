"""Run Stage 1 grid search"""
import json
import subprocess
from pathlib import Path

DATA_PATH = "data/sequences_continuous_balanced.parquet"
VOCAB_PATH = "data/vocab_continuous.json"
CONFIG_DIR = "configs"
STAGE = "weights"

config_files = sorted(Path(CONFIG_DIR).glob(f"{STAGE}_*.json"))
print(f"Found {len(config_files)} configs for {STAGE}")
print("=" * 50)

for i, config_file in enumerate(config_files, 1):
    print(f"\n[{i}/{len(config_files)}] Running: {config_file.name}")
    
    with open(config_file) as f:
        config = json.load(f)
    
    cmd = [
        "python", "train_xt_model.py",
        "--data_path", DATA_PATH,
        "--vocab_path", VOCAB_PATH,
        "--d_model", str(config['d_model']),
        "--num_layers", str(config['num_layers']),
        "--nhead", str(config['nhead']),
        "--n_components", str(config['n_components']),
        "--dropout", str(config['dropout']),
        "--epochs", str(config['epochs']),
        "--batch_size", str(config['batch_size']),
        "--lr", str(config['lr']),
        "--weight_decay", str(config['weight_decay']),
        "--weight_shot", str(config['weight_shot']),
        "--weight_goal", str(config['weight_goal']),
        "--weight_no_goal", str(config['weight_no_goal']),
        "--n_rollouts", str(config['n_rollouts']),
        "--run_name", config['run_name'],
        "--experiment_name", "Expected_Threat_TOP5_Stage1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Completed {i}/{len(config_files)}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "=" * 50)
print("Stage 1 completed!")