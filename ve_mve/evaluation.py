"""
evaluation.py - Simplified Evaluation Module
"""
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GPULogisticRegression(nn.Module):
    def __init__(self, input_dim, n_classes=2, base_ridge_lambda=0.01, reference_dim=512, lr=0.01, n_epochs=50, seed=42):
        super().__init__()
        
        set_seed(seed)
        
        self.n_classes = n_classes
        self.linear = nn.Linear(input_dim, n_classes)
        
        self.ridge_lambda = base_ridge_lambda * (reference_dim / input_dim)
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X_train, y_train):
        set_seed(self.seed)
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        if torch.min(y_train) < 0 or torch.max(y_train) >= self.n_classes:
            y_train = torch.clamp(y_train, 0, self.n_classes - 1)
        
        self.mean = X_train.mean(0)
        self.std = X_train.std(0) + 1e-8
        X_train_norm = (X_train - self.mean) / self.std
        
        self.to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            
            logits = self.forward(X_train_norm)
            loss = criterion(logits, y_train)
            
            ridge_penalty = 0
            for name, param in self.named_parameters():
                if 'weight' in name:
                    ridge_penalty += torch.norm(param, p=2)
            
            total_loss = loss + self.ridge_lambda * ridge_penalty
            
            total_loss.backward()
            optimizer.step()
    
    def predict_proba(self, X_test):
        self.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(X_test).to(self.device)
            X_test_norm = (X_test - self.mean) / self.std
            
            logits = self.forward(X_test_norm)
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().numpy()
    
    def predict_proba_positive(self, X_test):
        if self.n_classes != 2:
            raise ValueError("predict_proba_positive only works for binary classification")
        probs = self.predict_proba(X_test)
        return probs[:, 1]


def train_predict(X_train, y_train, X_test, seed=42):
    n_classes = len(np.unique(y_train))
    model = GPULogisticRegression(X_train.shape[1], n_classes=n_classes, seed=seed)
    model.fit(X_train, y_train)
    
    if n_classes == 2:
        return model.predict_proba_positive(X_test)
    else:
        return model.predict_proba(X_test)


def find_optimal_threshold_torch(y_true, y_pred_proba, device):
    y_true = torch.FloatTensor(y_true).to(device)
    y_pred_proba = torch.FloatTensor(y_pred_proba).to(device)
    
    thresholds = torch.linspace(0.01, 0.99, 99).to(device)
    best_f1 = 0
    best_threshold = torch.tensor(0.5).to(device)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).float()
        
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold.cpu().item()


def calculate_metrics_torch(y_true, y_pred_proba, threshold, device, is_binary=True):
    y_true = torch.FloatTensor(y_true).to(device)
    
    if is_binary:
        y_pred_proba = torch.FloatTensor(y_pred_proba).to(device)
        y_pred = (y_pred_proba > threshold).float()
        
        sorted_indices = torch.argsort(y_pred_proba, descending=True)
        y_true_sorted = y_true[sorted_indices]
        
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            auc = 0.5
        else:
            cum_pos = torch.cumsum(y_true_sorted, dim=0)
            cum_neg = torch.cumsum(1 - y_true_sorted, dim=0)
            
            tpr = cum_pos / n_pos
            fpr = cum_neg / n_neg
            
            auc = torch.trapz(tpr, fpr).item()
        
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        tn = ((1 - y_pred) * (1 - y_true)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / len(y_true)
        
    else:
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = torch.FloatTensor(y_pred_proba).to(device)
        else:
            y_pred_proba = torch.FloatTensor(y_pred_proba).to(device)
        
        if len(y_pred_proba.shape) == 2:
            y_pred = torch.argmax(y_pred_proba, dim=1).float()
        else:
            y_pred = (y_pred_proba > 0.5).float()
        
        accuracy = (y_pred == y_true).float().mean()
        
        auc = float(accuracy.cpu())
        f1 = float(accuracy.cpu())
    
    return {
        'auc': float(auc),
        'f1': float(f1.cpu()) if torch.is_tensor(f1) else float(f1),
        'accuracy': float(accuracy.cpu()) if torch.is_tensor(accuracy) else float(accuracy)
    }


def get_subgroup_indices(studies, demographics, include_subgroups=None):
    if include_subgroups is None:
        include_subgroups = ['overall']
    
    subgroups = {}
    
    if 'overall' in include_subgroups:
        subgroups['overall'] = list(range(len(studies)))
    
    for subgroup in include_subgroups:
        if subgroup not in subgroups:
            subgroups[subgroup] = []
    
    # View count categorization rules
    view_categories = {
        'view3': {
            'view3_low': lambda x: x < 30,
            'view3_medium': lambda x: 30 <= x <= 50,
            'view3_high': lambda x: x > 50
        },
        'view4': {
            'view4_lt20': lambda x: x < 20,
            'view4_20to39': lambda x: 20 <= x < 40,
            'view4_40to59': lambda x: 40 <= x < 60,
            'view4_ge60': lambda x: x >= 60
        }
    }
    
    for idx, study_id in enumerate(studies):
        demo = demographics.get(study_id, {})
        
        # Gender subgroups
        if 'male' in include_subgroups and demo.get('sex') == 'Male':
            subgroups['male'].append(idx)
        if 'female' in include_subgroups and demo.get('sex') == 'Female':
            subgroups['female'].append(idx)
        
        # Race subgroups
        if 'white' in include_subgroups and demo.get('race') == 'White':
            subgroups['white'].append(idx)
        elif 'black' in include_subgroups and demo.get('race') == 'Black':
            subgroups['black'].append(idx)
        elif 'others' in include_subgroups and demo.get('race') == 'Others':
            subgroups['others'].append(idx)
        
        # View count subgroups (dynamic categorization)
        view_cnt = demo.get('view_cnt', 0)
        
        # Check all view categories
        for category_group, categories in view_categories.items():
            for category_name, condition in categories.items():
                if category_name in include_subgroups and condition(view_cnt):
                    subgroups[category_name].append(idx)
                    
    return subgroups


def evaluate_with_subgroups(embeddings, labels, cv_splits, task, demographics, 
                           include_subgroups=None, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_results = []
    
    for fold_idx, split in enumerate(cv_splits):
        fold_seed = seed + fold_idx
        set_seed(fold_seed)
        
        train_studies = [i for i in split['train'] 
                        if i in embeddings and i in labels and task in labels[i]]
        test_studies = [i for i in split['test'] 
                       if i in embeddings and i in labels and task in labels[i]]
        
        if len(train_studies) == 0 or len(test_studies) == 0:
            continue
        
        X_train = np.stack([embeddings[i] for i in train_studies])
        y_train = [labels[i][task] for i in train_studies]
        
        y_train_clean = []
        X_train_clean = []
        for i, label in enumerate(y_train):
            if label is not None and not np.isnan(label):
                y_train_clean.append(int(label))
                X_train_clean.append(X_train[i])
        
        if len(y_train_clean) == 0:
            continue
            
        X_train = np.stack(X_train_clean)
        y_train = y_train_clean
        
        X_test = np.stack([embeddings[i] for i in test_studies])
        y_test = [labels[i][task] for i in test_studies]
        
        y_test_clean = []
        X_test_clean = []
        test_studies_clean = []
        for i, label in enumerate(y_test):
            if label is not None and not np.isnan(label):
                y_test_clean.append(int(label))
                X_test_clean.append(X_test[i])
                test_studies_clean.append(test_studies[i])
        
        if len(y_test_clean) == 0:
            continue
            
        X_test = np.stack(X_test_clean)
        y_test = y_test_clean
        test_studies = test_studies_clean
        
        unique_labels = np.unique(y_train + y_test)
        n_classes = len(unique_labels)
        is_binary = n_classes == 2
        
        model = GPULogisticRegression(X_train.shape[1], n_classes=n_classes, seed=fold_seed)
        model.fit(X_train, y_train)
        
        if is_binary:
            train_probs = model.predict_proba_positive(X_train)
            test_probs = model.predict_proba_positive(X_test)
            
            train_probs = np.array(train_probs)
            train_labels = np.array(y_train)
            test_probs = np.array(test_probs)
            test_labels = np.array(y_test)
            
            optimal_threshold = find_optimal_threshold_torch(train_labels, train_probs, device)
        else:
            test_probs = model.predict_proba(X_test)
            test_labels = np.array(y_test)
            optimal_threshold = 0.5
        
        subgroups = get_subgroup_indices(test_studies, demographics, include_subgroups)
        
        for subgroup_name, indices in subgroups.items():
            if len(indices) == 0:
                continue
                
            subgroup_labels = test_labels[indices]
            if is_binary:
                subgroup_probs = test_probs[indices]
            else:
                subgroup_probs = test_probs[indices]
            
            metrics = calculate_metrics_torch(subgroup_labels, subgroup_probs, 
                                            optimal_threshold, device, is_binary)
            
            metrics.update({
                'fold': fold_idx,
                'task': task,
                'subgroup': subgroup_name,
                'n_samples': len(indices),
                'n_positive': int(subgroup_labels.sum()) if is_binary else len(indices),
                'threshold': float(optimal_threshold),
                'seed': fold_seed,
                'n_classes': n_classes,
                'is_binary': is_binary
            })
            
            all_results.append(metrics)
    
    return all_results


def evaluate_multiple_tasks_with_subgroups(embeddings, labels, cv_splits, clinical_tasks, 
                                         demographic_tasks, demographics, all_subgroups=None, seed=42):
    import pandas as pd
    
    set_seed(seed)
    
    all_results = []
    
    # Default subgroups if not provided
    if all_subgroups is None:
        all_subgroups = [
            'overall', 'male', 'female', 'white', 'black', 'others',
            'view3_low', 'view3_medium', 'view3_high',
            'view4_lt20', 'view4_20to39', 'view4_40to59', 'view4_ge60'
        ]
    
    for task in clinical_tasks:
        task_results = evaluate_with_subgroups(embeddings, labels, cv_splits, task, 
                                             demographics, all_subgroups, seed)
        all_results.extend(task_results)
    
    for task in demographic_tasks:
        task_results = evaluate_with_subgroups(embeddings, labels, cv_splits, task, 
                                             demographics, ['overall'], seed)
        all_results.extend(task_results)
    
    return pd.DataFrame(all_results)