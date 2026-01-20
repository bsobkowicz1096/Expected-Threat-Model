"""
Loss Weights Optimization Configuration Generator
"""
import json
import itertools
from pathlib import Path

def generate_weight_configs():
    """Optimize loss weights around current best"""
    grid = {
        'weight_goal': [25, 30, 35],
        'weight_shot': [7, 10, 12],
        'weight_no_goal': [1.5, 2.0, 2.5]
    }
    
    # Fixed params (best from testing)
    fixed = {
        'd_model': 512,
        'num_layers': 12,
        'nhead': 8,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'n_components': 5,
        'epochs': 10,
        'batch_size': 32,
        'n_rollouts': 100,
    }
    
    configs = []
    for w_goal, w_shot, w_no_goal in itertools.product(*grid.values()):
        config = fixed.copy()
        config.update({
            'weight_goal': w_goal,
            'weight_shot': w_shot,
            'weight_no_goal': w_no_goal,
            'run_name': f'weights_g{w_goal}_s{w_shot}_n{w_no_goal}'
        })
        configs.append(config)
    
    return configs

def save_configs(configs, stage_name, output_dir='configs'):
    Path(output_dir).mkdir(exist_ok=True)
    
    for i, config in enumerate(configs):
        filename = f"{output_dir}/{stage_name}_{i:03d}.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ Saved {len(configs)} configs to {output_dir}/")

if __name__ == "__main__":
    configs = generate_weight_configs()
    print(f"Weight optimization: {len(configs)} configs")
    save_configs(configs, 'weights')