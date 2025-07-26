"""
data_processing.py - Simplified Data Processing
"""
import sys
sys.path.insert(0, "mve-echo/ve_mve")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple
from collections import defaultdict
import pickle
from pathlib import Path
from config import get_data_config, get_paths

class EchoDataProcessor:
    
    def __init__(self):
        self.config = get_data_config()
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_and_process_data(self, ve_path: str, outcome_path: str) -> Dict:
        cache_key = f"{Path(ve_path).stem}_{Path(outcome_path).stem}"
        cache_file = self.cache_dir / f"processed_data_{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        ve_df = pd.read_csv(ve_path)
        outcome_df = pd.read_csv(outcome_path)
        
        merged_df = pd.merge(outcome_df, ve_df, on='dicom_path', how='inner')
        
        if len(merged_df) == 0:
            raise ValueError("No data after merge")
        
        merged_df = self._process_demographics(merged_df)
        processed_data = self._create_study_data(merged_df)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        return processed_data
    
    def _process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['Sex_encoded'] = df['Sex_inUKN'].map(self.config['sex_mapping'])
        df['Race_encoded'] = df['Race_inUKN'].map(self.config['race_mapping'])
       
        return df
    
    def _create_study_data(self, df: pd.DataFrame) -> Dict:
        study_data = {}
        study_outcomes = {}
        study_demographics = {}
        subject_to_studies = defaultdict(list)
        
        for study_id, group in df.groupby('study_id'):
            ve_features = group[self.config['ve_columns']].values
            
            outcomes = {}
            for task in self.config['binary_tasks']:
                if task in group.columns:
                    value = group[task].iloc[0]
                    if pd.notna(value):
                        outcomes[task] = int(float(value))
            
            demographics = {
                'subject_id': group['subject_id'].iloc[0],
                'sex_encoded': int(group['Sex_encoded'].iloc[0]),
                'race_encoded': int(group['Race_encoded'].iloc[0]),
                'sex_label': group['Sex_inUKN'].iloc[0],
                'race_label': group['Race_inUKN'].iloc[0],
                'view_cnt': int(group['view_cnt'].iloc[0])
            }

            study_data[study_id] = {
                've_features': ve_features,
                'n_views': len(ve_features)
            }
            study_outcomes[study_id] = outcomes
            study_demographics[study_id] = demographics
            
            subject_to_studies[demographics['subject_id']].append(study_id)
                
        return {
            'study_data': study_data,
            'study_outcomes': study_outcomes,
            'study_demographics': study_demographics,
            'subject_to_studies': dict(subject_to_studies)
        }


class EchoDataset(Dataset):
    
    def __init__(self, study_data: Dict, study_demographics: Dict, 
                 study_ids: List[str], max_views: int = 128):
        self.study_data = study_data
        self.study_demographics = study_demographics
        self.study_ids = sorted(study_ids)
        self.max_views = max_views
    
    def __len__(self):
        return len(self.study_ids)
    
    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        
        ve_features = self.study_data[study_id]['ve_features']
        n_views = min(len(ve_features), self.max_views)
        
        padded_features = np.zeros((self.max_views, 512), dtype=np.float32)
        padded_features[:n_views] = ve_features[:n_views]
        
        view_mask = np.zeros(self.max_views, dtype=bool)
        view_mask[:n_views] = True
        
        demographics = self.study_demographics[study_id]
        
        return {
            'study_id': study_id,
            'features': torch.from_numpy(padded_features),
            'view_mask': torch.from_numpy(view_mask),
            'sex_encoded': torch.tensor(demographics['sex_encoded'], dtype=torch.long),
            'race_encoded': torch.tensor(demographics['race_encoded'], dtype=torch.long),
            'view_cnt': torch.tensor(demographics['view_cnt'], dtype=torch.long) 
        }


class DataSplitter:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def create_mae_train_val_evaluation_split(self, processed_data: Dict, 
                                             evaluation_ratio: float = 0.5,
                                             validation_ratio: float = 0.15,
                                             random_state: int = None) -> Tuple[List[str], List[str], List[str]]:
        
        subject_to_studies = processed_data['subject_to_studies']
        all_subjects = sorted(list(subject_to_studies.keys()))
        
        seed = random_state if random_state is not None else self.random_state
        np.random.seed(seed)
        np.random.shuffle(all_subjects)
        
        n_eval_subjects = int(len(all_subjects) * evaluation_ratio)
        eval_subjects = all_subjects[:n_eval_subjects]
        remaining_subjects = all_subjects[n_eval_subjects:]
        
        n_val_subjects = int(len(remaining_subjects) * validation_ratio)
        val_subjects = remaining_subjects[:n_val_subjects]
        train_subjects = remaining_subjects[n_val_subjects:]
        
        mae_train_studies = []
        mae_val_studies = []
        evaluation_studies = []
        
        for subject in train_subjects:
            mae_train_studies.extend(subject_to_studies[subject])
        
        for subject in val_subjects:
            mae_val_studies.extend(subject_to_studies[subject])
        
        for subject in eval_subjects:
            evaluation_studies.extend(subject_to_studies[subject])
        
        mae_train_studies = sorted(mae_train_studies)
        mae_val_studies = sorted(mae_val_studies)
        evaluation_studies = sorted(evaluation_studies)
        
        return mae_train_studies, mae_val_studies, evaluation_studies
    
    def create_cv_splits(self, studies: List[str], subject_to_studies: Dict, 
                        n_folds: int = 3, random_state: int = None) -> List[Dict]:
        
        study_to_subject = {}
        for subject, study_list in subject_to_studies.items():
            for study in study_list:
                study_to_subject[study] = subject
        
        subjects = sorted(list(set(study_to_subject[s] for s in studies if s in study_to_subject)))
        
        seed = random_state if random_state is not None else self.random_state
        np.random.seed(seed)
        
        gkf = GroupKFold(n_splits=n_folds)
        dummy_y = np.zeros(len(subjects))
        
        splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(subjects, dummy_y, groups=subjects)):
            train_subjects = [subjects[i] for i in train_idx]
            test_subjects = [subjects[i] for i in test_idx]
            
            train_studies = []
            test_studies = []
            
            for subject in train_subjects:
                train_studies.extend([s for s in subject_to_studies[subject] if s in studies])
            
            for subject in test_subjects:
                test_studies.extend([s for s in subject_to_studies[subject] if s in studies])
            
            train_studies = sorted(train_studies)
            test_studies = sorted(test_studies)
            
            splits.append({
                'fold': fold_idx,
                'train': train_studies,
                'test': test_studies
            })
        
        return splits


def create_baseline_embeddings(processed_data: Dict) -> Dict[str, np.ndarray]:
    cache_file = Path("cache/foundation_embeddings.pkl")
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    paths = get_paths()
    foundation_df = pd.read_csv(paths['ve7680_data_path'])
    
    ve_columns = [col for col in foundation_df.columns 
                 if col.startswith('ve') and col[2:].isdigit()]
    ve_columns = sorted(ve_columns, key=lambda x: int(x[2:]))
    
    if len(ve_columns) < 1000:
        return _create_statistical_baseline_embeddings(processed_data)
    
    study_id_col = None
    for col in ['study_id', 'StudyID', 'STUDY_ID', 'id']:
        if col in foundation_df.columns:
            study_id_col = col
            break
    
    if study_id_col is None:
        return _create_statistical_baseline_embeddings(processed_data)

    embeddings = {}
    for _, row in foundation_df.iterrows():
        study_id = str(row[study_id_col])
        if study_id in processed_data['study_data']:
            ve_embedding = row[ve_columns].values.astype(np.float32)
            
            if np.isfinite(ve_embedding).all():
                embeddings[study_id] = ve_embedding
    
    if len(embeddings) == 0:
        return _create_statistical_baseline_embeddings(processed_data)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings


def _create_statistical_baseline_embeddings(processed_data: Dict) -> Dict[str, np.ndarray]:
    cache_file = Path("cache/statistical_embeddings.pkl")
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    embeddings = {}
    
    for study_id, study_info in processed_data['study_data'].items():
        ve_features = study_info['ve_features']
        
        mean_features = np.mean(ve_features, axis=0)
        std_features = np.std(ve_features, axis=0)
        max_features = np.max(ve_features, axis=0)
        min_features = np.min(ve_features, axis=0)
        p25_features = np.percentile(ve_features, 25, axis=0)
        p75_features = np.percentile(ve_features, 75, axis=0)
        range_features = max_features - min_features
        
        combined_features = np.concatenate([
            mean_features, std_features, max_features, min_features,
            p25_features, p75_features, range_features
        ])
        
        embeddings[study_id] = combined_features.astype(np.float32)
    
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings