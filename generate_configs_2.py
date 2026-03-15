"""
Generate configs for Stage 2 - Hyperparameter optimization
"""
import json
from pathlib import Path
from itertools import product

# Output directory
OUTPUT_DIR = Path("configs_2")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed parameters from Stage 1 optimal weights
fixed = {
    'd_model': 512,
    'num_layers': 12,
    'nhead': 8,
    'dropout': 0.1,
    'n_components': 5,
    'epochs': 12,
    'batch_size': 32,
    'weight_decay': 0.01,
    'n_rollouts': 100,
    # Optimal weights from Stage 1
    'weight_goal': 15.0,
    'weight_shot': 3.0,
    'weight_no_goal': 1.5,
}

# Grid parameters to optimize
grid = {
    'lr': [5e-5, 1e-4, 2e-4],
    'loss_ratio': [0.3, 0.5, 1.0, 2.0, 3.0],
}

# Generate all combinations
configs = []
for lr, loss_ratio in product(grid['lr'], grid['loss_ratio']):
    config = fixed.copy()
    config['lr'] = lr
    config['loss_ratio'] = loss_ratio
    
    # Generate run name
    lr_str = f"{lr:.0e}".replace('e-0', 'e-').replace('+', '')
    ratio_str = f"{loss_ratio:.1f}".replace('.', '_')
    run_name = f"hyperparams_lr{lr_str}_r{ratio_str}"
    config['run_name'] = run_name
    
    configs.append(config)

# Save configs
for config in configs:
    filename = f"{config['run_name']}.json"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

print(f"Generated {len(configs)} configs in {OUTPUT_DIR}/")
print("\nGrid:")
for key, values in grid.items():
    print(f"  {key}: {values}")
print(f"\nTotal experiments: {len(configs)}")
