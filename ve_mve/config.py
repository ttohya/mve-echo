"""
config.py - Simplified Configuration
"""
import torch
import numpy as np
from pathlib import Path

result_dir = # the path to the result dir
ds_dir = # dataset dir which includes "ve_echoprime.csv" and "echo_ds.csv")

CONFIG = {
    'model': {
        'input_dim': 512,
        'embed_dim': 512,
        'num_layers': 2,
        'num_heads': 8,
        'max_views': 128,
        'mask_ratio': 0.50,
        'use_adversarial': True
    },
    'training': {
        'mae_epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        
        'early_stopping': {
            'patience': 10,
            'min_delta': 1e-4,
            'warmup_epochs': 5,
            'restore_best_weights': True
        }
    },
    'evaluation': {
        'n_folds': 4,
        'n_repeats': 5,
        'adversarial_weights': [-1, 0, 0.1, 0.2, 0.5, 1.0],
        'test_ratio': 0.5,
        'validation_ratio': 0.15,
        'gpu_batch_size': 16,
        'gpu_epochs': 30,
        'use_gpu_evaluation': True
    },
    'data': {
        've_columns': [f've{i:03d}' for i in range(1, 513)],
        'binary_tasks': [
            'ab_Ao', 'ab_LVD', 'ab_LAVI', 'ab_LVMI', 'rEF', 'RV_func',
            'mildAR', 'mildAS', 'mildMR', 'mildTR', 'modAR', 'modAS',
            'modMR', 'modTR', 'PHT', 'ab_E_e', 'IVC','Dilated_cardiompyopthy',
            'Hypertrophic_cardiomyopathy', 'Pulmonary_embolism', 'Myocardial_infarction'
        ],
        'demo_tasks': [
            'sex_encoded', 'race_encoded'
        ],
        'sex_mapping': {'Female': 0, 'Male': 1, 'Unknown': 2},
        'race_mapping': {'White': 0, 'Black': 1, 'Others': 2, 'Unknown': 3},
        
        # View count categorization rules (based on view_cnt)
        'view_categories': {
            'view3': {  # 3 categories for view counts (Low/Medium/High)
                'low': lambda x: x < 30,
                'medium': lambda x: 30 <= x <= 50,
                'high': lambda x: x > 50
            },
            'view4': {  # 4 categories for view counts
                'lt20': lambda x: x < 20,
                '20to39': lambda x: 20 <= x < 40,
                '40to59': lambda x: 40 <= x < 60,
                'ge60': lambda x: x >= 60
            }
        }
    },
    'paths': {
        've_data_path': ds_dir + "/ve_echoprime.csv",
        've7680_data_path': ds_dir + "/all_ve_echoprime_7680.csv",
        'outcome_data_path': ds_dir + "/echo_ds.csv",
        'output_dir': result_dir
    }
}

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return device


def get_config():
    return CONFIG.copy()

def get_model_config():
    return CONFIG['model'].copy()

def get_training_config():
    return CONFIG['training'].copy()

def get_evaluation_config():
    return CONFIG['evaluation'].copy()

def get_data_config():
    return CONFIG['data'].copy()

def get_paths():
    return CONFIG['paths'].copy()

def validate_paths():
    paths = get_paths()
    
    for name, path in paths.items():
        if name.endswith('_path'):
            path_obj = Path(path)
            if not path_obj.exists():
                return False
    
    return True

def create_output_dir(output_dir=None):
    if output_dir is None:
        output_dir = CONFIG['paths']['output_dir']
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    return output_path


if __name__ == "__main__":
    device = setup_device()
    
    paths_valid = validate_paths()
    if paths_valid:
        print("✅ All paths validated")
    else:
        print("❌ Path validation failed")