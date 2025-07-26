"""
training.py - Simplified MAE Training Module
"""
import sys
sys.path.insert(0, "mve-echo/ve_mve")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict

from config import get_training_config, get_model_config, get_evaluation_config
from models import create_mae_model
from data_processing import EchoDataset


class MAETrainer:
    def __init__(self, device: torch.device, adversarial_weight: float = 0.1):
        self.device = device
        self.adversarial_weight = adversarial_weight
        self.training_config = get_training_config()
        self.model_config = get_model_config()
        
        self.patience = self.training_config.get('early_stopping', {}).get('patience', 15)
        self.min_delta = self.training_config.get('early_stopping', {}).get('min_delta', 1e-4)
        self.warmup_epochs = self.training_config.get('early_stopping', {}).get('warmup_epochs', 5)
        self.restore_best_weights = self.training_config.get('early_stopping', {}).get('restore_best_weights', True)
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
    
    def setup_model(self, use_adversarial: bool = True):
        self.model = create_mae_model(
            input_dim=self.model_config['input_dim'],
            embed_dim=self.model_config['embed_dim'],
            num_layers=self.model_config['num_layers'],
            num_heads=self.model_config['num_heads'],
            max_views=self.model_config['max_views'],
            mask_ratio=self.model_config['mask_ratio'],
            use_adversarial=use_adversarial
        )
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'adversarial': 0.0
        }
        
        lambda_p = self._compute_lambda_p(epoch)
        n_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features'].to(self.device, non_blocking=True)
            view_mask = batch['view_mask'].to(self.device, non_blocking=True)
            sex_labels = batch['sex_encoded'].to(self.device, non_blocking=True)
            race_labels = batch['race_encoded'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                features, 
                view_mask=view_mask, 
                lambda_p=lambda_p, 
                training_mode='mae'
            )
            
            total_loss, losses = self._compute_losses(
                outputs, features, sex_labels, race_labels
            )
            
            if total_loss.item() < 100.0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['reconstruction'] += losses['reconstruction']
            epoch_losses['adversarial'] += losses['adversarial']
        
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        epoch_losses['lambda_p'] = lambda_p
        
        return epoch_losses
    
    def _compute_losses(self, outputs: Dict, features: torch.Tensor, 
                       sex_labels: torch.Tensor, race_labels: torch.Tensor) -> tuple:
        losses = {'reconstruction': 0.0, 'adversarial': 0.0}
        
        reconstruction_loss = torch.tensor(0.0, device=self.device)
        
        if 'reconstructed' in outputs and 'mask_indices' in outputs:
            mask_indices = outputs['mask_indices']
            
            if mask_indices.any():
                reconstructed = outputs['reconstructed']
                reconstruction_loss = F.mse_loss(
                    reconstructed[mask_indices], 
                    features[mask_indices]
                )
        
        total_loss = reconstruction_loss
        losses['reconstruction'] = reconstruction_loss.item()
        
        if self.adversarial_weight > 0 and 'adversarial_predictions' in outputs:
            adversarial_preds = outputs['adversarial_predictions']
            
            sex_loss = F.cross_entropy(adversarial_preds['sex'], sex_labels)
            race_loss = F.cross_entropy(adversarial_preds['race'], race_labels)
            
            adversarial_loss = sex_loss + race_loss
            total_loss += self.adversarial_weight * adversarial_loss
            losses['adversarial'] = adversarial_loss.item()
        
        return total_loss, losses
    
    def _compute_lambda_p(self, epoch: int) -> float:
        return 1.0 if self.adversarial_weight > 0 else 0.0

    def train_full_model_with_validation(self, processed_data: Dict, 
                                       train_study_ids: list, 
                                       val_study_ids: list) -> torch.nn.Module:
        
        print(f"Training MVE: {len(train_study_ids)} train, {len(val_study_ids)} val studies (adversarial_weight={self.adversarial_weight})")
        
        use_adversarial = self.adversarial_weight > 0
        self.setup_model(use_adversarial=use_adversarial)
        
        train_dataset = EchoDataset(
            processed_data['study_data'],
            processed_data['study_demographics'],
            train_study_ids,
            max_views=self.model_config['max_views']
        )
        
        val_dataset = EchoDataset(
            processed_data['study_data'],
            processed_data['study_demographics'],
            val_study_ids,
            max_views=self.model_config['max_views']
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        best_model_state = None
        
        for epoch in range(self.training_config['mae_epochs']):
            train_losses = self.train_epoch(train_dataloader, epoch)
            val_losses = self.validate_epoch(val_dataloader, epoch)
            
            current_val_loss = val_losses['total']
            improved = current_val_loss < (self.best_val_loss - self.min_delta)
            
            if improved:
                self.best_val_loss = current_val_loss
                self.early_stop_counter = 0
                self.best_epoch = epoch
                if self.restore_best_weights:
                    best_model_state = self.model.state_dict().copy()
                print(f"  Epoch {epoch:2d}: train={train_losses['total']:.4f} val={current_val_loss:.4f} adv_w={self.adversarial_weight:.1f} λ={train_losses['lambda_p']:.1f} ★")
            else:
                if epoch >= self.warmup_epochs:
                    self.early_stop_counter += 1
                print(f"  Epoch {epoch:2d}: train={train_losses['total']:.4f} val={current_val_loss:.4f} adv_w={self.adversarial_weight:.1f} λ={train_losses['lambda_p']:.1f} ({self.early_stop_counter}/{self.patience})")
            
            if epoch >= self.warmup_epochs and self.early_stop_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        if best_model_state is not None and self.restore_best_weights:
            self.model.load_state_dict(best_model_state)
            print(f"  Best model restored from epoch {self.best_epoch}")
        
        print(f"Training completed: adversarial_weight={self.adversarial_weight}")
        
        return self.model

    def validate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'adversarial': 0.0
        }
        
        lambda_p = self._compute_lambda_p(epoch)
        n_batches = len(dataloader)
        
        if n_batches == 0:
            return {
                'total': 999.0,
                'reconstruction': 999.0,
                'adversarial': 999.0
            }
        
        valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                features = batch['features'].to(self.device, non_blocking=True)
                view_mask = batch['view_mask'].to(self.device, non_blocking=True)
                sex_labels = batch['sex_encoded'].to(self.device, non_blocking=True)
                race_labels = batch['race_encoded'].to(self.device, non_blocking=True)
                
                if features.numel() == 0:
                    continue
                
                outputs = self.model(
                    features, 
                    view_mask=view_mask, 
                    lambda_p=lambda_p, 
                    training_mode='mae'
                )
                
                total_loss, losses = self._compute_losses(
                    outputs, features, sex_labels, race_labels
                )
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue
                
                epoch_losses['total'] += total_loss.item()
                epoch_losses['reconstruction'] += losses['reconstruction']
                epoch_losses['adversarial'] += losses['adversarial']
                
                valid_batches += 1
        
        if valid_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= valid_batches
        else:
            epoch_losses = {
                'total': 999.0,
                'reconstruction': 999.0,
                'adversarial': 999.0
            }
        
        return epoch_losses


def extract_embeddings(model: torch.nn.Module, processed_data: Dict, 
                      study_ids: list, device: torch.device) -> Dict[str, np.ndarray]:
    
    model.eval()
    
    dataset = EchoDataset(
        processed_data['study_data'],
        processed_data['study_demographics'],
        study_ids,
        max_views=get_model_config()['max_views']
    )
    
    evaluation_config = get_evaluation_config()
    dataloader = DataLoader(
        dataset,
        batch_size=evaluation_config['gpu_batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    embeddings = {}
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device, non_blocking=True)
            view_mask = batch['view_mask'].to(device, non_blocking=True)
            study_ids_batch = batch['study_id']
            
            outputs = model(features, view_mask=view_mask, training_mode='inference')
            
            batch_embeddings = outputs['embedding'].cpu().numpy()
            
            for i, study_id in enumerate(study_ids_batch):
                embeddings[study_id] = batch_embeddings[i]
    
    return embeddings