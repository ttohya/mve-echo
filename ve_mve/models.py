"""
models.py - Simplified MAE Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GradientReversal(nn.Module):
    
    def __init__(self, lambda_p: float = 1.0):
        super().__init__()
        self.lambda_p = lambda_p
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_p)
    
    def set_lambda(self, lambda_p: float):
        self.lambda_p = lambda_p


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_p, None


class FastSharedTransformer(nn.Module):
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 12, num_layers: int = 3):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        return self.transformer(x)


class FastAttentionAggregator(nn.Module):
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        torch.manual_seed(42)
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        query = self.global_query.expand(batch_size, 1, self.embed_dim)
        key_mask = ~mask if mask is not None else None
        
        out, _ = self.attention(query, x, x, key_padding_mask=key_mask)
        return self.layer_norm(out.squeeze(1))


class Discriminator(nn.Module):
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.gradient_reversal = GradientReversal()
        
        self.sex_classifier = nn.Linear(embed_dim, 3)  # F, M, Unknown
        self.race_classifier = nn.Linear(embed_dim, 4)  # White, Black, Others, Unknown
    
    def forward(self, x, lambda_p=0.1):
        x_rev = self.gradient_reversal(x)
        
        return {
            'sex': self.sex_classifier(x_rev),
            'race': self.race_classifier(x_rev)
        }

class EchoMAE(nn.Module):
    
    def __init__(self, 
                 input_dim: int = 512,
                 embed_dim: int = 512, 
                 num_layers: int = 3,
                 num_heads: int = 12,
                 max_views: int = 128,
                 mask_ratio: float = 0.50,
                 use_adversarial: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_views = max_views
        self.mask_ratio = mask_ratio
        self.use_adversarial = use_adversarial
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.shared_transformer = FastSharedTransformer(embed_dim, num_heads, num_layers)
        self.aggregator = FastAttentionAggregator(embed_dim, num_heads)
        self.recon_head = nn.Linear(embed_dim, input_dim)
        torch.manual_seed(42)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if use_adversarial:
            self.discriminator = Discriminator(embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, view_mask=None, lambda_p=0.1, training_mode='inference'):
        batch_size, max_views, _ = x.shape
        device = x.device
        
        x = self.input_proj(x)
        
        x_flat = x.view(batch_size * max_views, 1, self.embed_dim)
        
        if view_mask is not None:
            mask_flat = view_mask.view(-1)
        else:
            mask_flat = torch.ones(batch_size * max_views, dtype=torch.bool, device=device)
        
        mask_indices = None
        if training_mode == 'mae':
            x_flat, mask_indices = self._apply_random_masking(x_flat, mask_flat, device)
        
        encoded_flat = self.shared_transformer(x_flat)
        
        encoded = encoded_flat.view(batch_size, max_views, self.embed_dim)
        
        study_embedding = self.aggregator(encoded, view_mask)
        
        outputs = {'embedding': study_embedding}
        
        if self.training and training_mode == 'mae':
            recon = self.recon_head(study_embedding)
            outputs['reconstructed'] = recon.unsqueeze(1).expand(-1, max_views, -1)
            
            if mask_indices is not None:
                outputs['mask_indices'] = mask_indices
            else:
                outputs['mask_indices'] = torch.zeros(batch_size, max_views, dtype=torch.bool, device=device)
            
            if self.use_adversarial:
                outputs['adversarial_predictions'] = self.discriminator(study_embedding, lambda_p)
        
        return outputs
    
    def _apply_random_masking(self, x_flat, mask_flat, device):
        total_elements = x_flat.shape[0]
        
        should_mask = torch.rand(total_elements, device=device) < self.mask_ratio
        should_mask = should_mask & mask_flat
        
        if should_mask.any():
            x_flat = x_flat.clone()
            mask_tokens = self.mask_token.expand(should_mask.sum(), 1, self.embed_dim).to(device)
            x_flat[should_mask] = mask_tokens
        
        batch_size = total_elements // self.max_views
        mask_indices = should_mask.view(batch_size, self.max_views)
        
        return x_flat, mask_indices
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mae_model(**kwargs):
    from config import get_model_config  # configから取得するように修正
    
    set_reproducible_mode()
    
    # configのデフォルト値を使用
    default_config = get_model_config()
    default_config.update(kwargs)
    
    model = EchoMAE(**default_config)
    return model


def set_reproducible_mode():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_reproducible_mode()
    model = create_mae_model()
    
    batch_size, max_views, input_dim = 8, 128, 512
    x = torch.randn(batch_size, max_views, input_dim)
    mask = torch.ones(batch_size, max_views, dtype=torch.bool)
    
    model.train()
    with torch.no_grad():
        outputs1 = model(x, mask, training_mode='mae')
        outputs2 = model(x, mask, training_mode='mae')
        
        mask1 = outputs1.get('mask_indices', torch.zeros(1))
        mask2 = outputs2.get('mask_indices', torch.zeros(1))
        
        if torch.equal(mask1, mask2):
            print("Warning: Masks are identical")
        else:
            print("Random masking working correctly")