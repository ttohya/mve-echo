"""
step00_main.py
"""
import sys
sys.path.insert(0, "mve-echo/ve_mve")
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle
import random
from config import setup_device, get_config, get_paths, create_output_dir, validate_paths, get_evaluation_config
from data_processing import EchoDataProcessor, DataSplitter, create_baseline_embeddings
from training import MAETrainer, extract_embeddings
from evaluation import evaluate_multiple_tasks_with_subgroups


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMAEPipeline:
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        set_global_seed(base_seed)
        
        self.config = get_config()
        self.device = setup_device()
        self.processor = EchoDataProcessor()
        self.splitter = DataSplitter()
        
    def run_experiment(self, adversarial_weight: float, repeat_idx: int, 
                      processed_data: dict, evaluation_only: bool = False) -> dict:
        
        repeat_seed = self.base_seed + repeat_idx
        set_global_seed(repeat_seed)
        
        eval_config = get_evaluation_config()
        train_studies, val_studies, test_studies = self.splitter.create_mae_train_val_evaluation_split(
            processed_data, 
            evaluation_ratio=eval_config['test_ratio'],
            validation_ratio=eval_config['validation_ratio'],
            random_state=repeat_seed
        )
        
        if adversarial_weight >= 0:
            if evaluation_only:
                embeddings = self._load_embeddings(adversarial_weight, repeat_idx)
            else:
                embeddings = self._train_and_extract(
                    adversarial_weight, processed_data, train_studies, val_studies, test_studies, repeat_seed
                )
                self._save_embeddings(embeddings, adversarial_weight, repeat_idx)
        else:
            all_baseline = create_baseline_embeddings(processed_data)
            embeddings = {sid: all_baseline[sid] for sid in test_studies if sid in all_baseline}
        
        cv_splits = self.splitter.create_cv_splits(
            test_studies, processed_data['subject_to_studies'], 
            n_folds=self.config['evaluation']['n_folds'], random_state=repeat_seed
        )
        
        results = self._evaluate_embeddings(
            embeddings, processed_data, cv_splits, repeat_seed
        )
        
        return results
    
    def _train_and_extract(self, adversarial_weight: float, processed_data: dict, 
                          train_studies: list, val_studies: list, test_studies: list, seed: int) -> dict:
        set_global_seed(seed)
        
        trainer = MAETrainer(device=self.device, adversarial_weight=adversarial_weight)
        model = trainer.train_full_model_with_validation(processed_data, train_studies, val_studies)
        
        embeddings = extract_embeddings(model, processed_data, test_studies, self.device)
        
        return embeddings
    
    def _evaluate_embeddings(self, mae_embeddings: dict, 
                           processed_data: dict, cv_splits: list, seed: int) -> dict:
        set_global_seed(seed)
        
        clinical_tasks = self.config['data']['binary_tasks']
        demographic_tasks = self.config['data']['demo_tasks']

        demographics = {}
        
        for study_id, demo in processed_data['study_demographics'].items():
            view_cnt = demo.get('view_cnt', 0)
            
            if view_cnt < 30:
                view3 = 'Low'
            elif view_cnt <= 50:
                view3 = 'Medium'
            else:
                view3 = 'High'
            
            demographics[study_id] = {
                'sex': demo['sex_label'],
                'race': demo['race_label'],
                'view': view3,
                'view_cnt': view_cnt
            }
        
        labels = {}
        for study_id in processed_data['study_data'].keys():
            labels[study_id] = {}
            
            outcomes = processed_data['study_outcomes'].get(study_id, {})
            for task in clinical_tasks:
                if task in outcomes:
                    labels[study_id][task] = outcomes[task]
            
            demo = processed_data['study_demographics'].get(study_id, {})

            sex_encoded = demo.get('sex_encoded', 2)
            if sex_encoded in [0, 1]:
                labels[study_id]['sex_encoded'] = sex_encoded

            race_encoded = demo.get('race_encoded', 3)
            if race_encoded in [0, 1, 2]:
                labels[study_id]['race_encoded'] = race_encoded

        results = {}
        
        all_subgroups = [
            'overall', 'male', 'female', 'white', 'black', 'others',
            'view3_low', 'view3_medium', 'view3_high',
            'view4_lt20', 'view4_20to39', 'view4_40to59', 'view4_ge60'
        ]
        
        mae_results = evaluate_multiple_tasks_with_subgroups(
            mae_embeddings, labels, cv_splits, clinical_tasks, demographic_tasks, 
            demographics, all_subgroups=all_subgroups, seed=seed
        )
        results['mae_ve'] = mae_results
        
        return results
    
    def _save_embeddings(self, embeddings: dict, weight: float, repeat: int):
        output_dir = Path(self.config['paths']['output_dir']) / "artifacts/embeddings"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"mae_embeddings_w{weight}_r{repeat}.pkl"
        with open(output_dir / filename, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def _load_embeddings(self, weight: float, repeat: int) -> dict:
        output_dir = Path(self.config['paths']['output_dir']) / "artifacts/embeddings"
        filename = f"mae_embeddings_w{weight}_r{repeat}.pkl"
        
        with open(output_dir / filename, 'rb') as f:
            return pickle.load(f)
    
    def run_full_experiment(self, evaluation_only: bool = False, n_repeats: int = None) -> str:
        print(f"Simple MAE Pipeline (seed={self.base_seed})")
        
        if not validate_paths():
            raise RuntimeError("Data path validation failed")
        
        n_repeats = self.config['evaluation']['n_repeats']
        
        paths = get_paths()
        processed_data = self.processor.load_and_process_data(
            paths['ve_data_path'], paths['outcome_data_path']
        )
        
        weights = self.config['evaluation']['adversarial_weights']
        
        all_results = []
        
        for weight in weights:
            for repeat in range(n_repeats):
                result = self.run_experiment(
                    weight, repeat, processed_data, evaluation_only
                )
                
                mae_df = result['mae_ve'].copy()
                if weight < 0:
                    mae_df['model_type'] = 'Foundation_VE'
                else:   
                    mae_df['model_type'] = f'MVE_VE_w{weight}'
                mae_df['adversarial_weight'] = weight
                mae_df['repeat_idx'] = repeat
                mae_df['global_seed'] = self.base_seed + repeat
                
                all_results.append(mae_df)
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            
            output_path = Path(self.config['paths']['output_dir'])
            output_path.mkdir(exist_ok=True, parents=True)
            csv_file = output_path / "mve_evaluation_results.csv"
            final_df.to_csv(csv_file, index=False)
            
            print(f"Results saved to: {csv_file}")
            
            return str(csv_file)
        else:
            return ""


def main(evaluation_only: bool = False, n_repeats: int = None, base_seed: int = 42):
    pipeline = SimpleMAEPipeline(base_seed=base_seed)
    csv_path = pipeline.run_full_experiment(evaluation_only=evaluation_only, n_repeats=n_repeats)
    return csv_path


if __name__ == "__main__":
    csv_path = main(evaluation_only=False) 