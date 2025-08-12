"""
Fusion model for combining multimodal features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SimpleFusion(nn.Module):
    """
    Simple concatenation-based fusion
    """
    
    def __init__(self, config):
        super(SimpleFusion, self).__init__()
        
        self.config = config
        self.input_dim = config.fusion.input_dim
        self.dropout = config.fusion.dropout
        
        # Hidden dimensions for fusion layers
        if config.fusion.hidden_dims is None:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = config.fusion.hidden_dims
        
        # Build fusion layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_layers = nn.Sequential(*layers)
    
    def forward(self, visual_embedding: torch.Tensor, 
                audio_embedding: torch.Tensor, 
                temporal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for simple fusion
        
        Args:
            visual_embedding: Visual features (batch_size, visual_dim)
            audio_embedding: Audio features (batch_size, audio_dim)
            temporal_embedding: Temporal features (batch_size, temporal_dim)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # Concatenate all embeddings
        fused_features = torch.cat([visual_embedding, audio_embedding, temporal_embedding], dim=1)
        
        # Apply fusion layers
        logits = self.fusion_layers(fused_features)
        
        return logits


class AttentionFusion(nn.Module):
    """
    Fusion with attention mechanism
    """
    
    def __init__(self, config):
        super(AttentionFusion, self).__init__()
        
        self.config = config
        self.visual_dim = config.visual.embedding_dim
        self.audio_dim = config.audio.embedding_dim
        self.temporal_dim = config.temporal.embedding_dim
        self.dropout = config.fusion.dropout
        self.attention_heads = config.fusion.attention_heads
        
        # Project all modalities to same dimension
        self.projection_dim = 256
        self.visual_projection = nn.Linear(self.visual_dim, self.projection_dim)
        self.audio_projection = nn.Linear(self.audio_dim, self.projection_dim)
        self.temporal_projection = nn.Linear(self.temporal_dim, self.projection_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.projection_dim,
            num_heads=self.attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # Fusion layers
        if config.fusion.hidden_dims is None:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = config.fusion.hidden_dims
        
        layers = []
        prev_dim = self.projection_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_layers = nn.Sequential(*layers)
    
    def forward(self, visual_embedding: torch.Tensor, 
                audio_embedding: torch.Tensor, 
                temporal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention fusion
        
        Args:
            visual_embedding: Visual features (batch_size, visual_dim)
            audio_embedding: Audio features (batch_size, audio_dim)
            temporal_embedding: Temporal features (batch_size, temporal_dim)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        batch_size = visual_embedding.size(0)
        
        # Project all modalities to same dimension
        visual_proj = self.visual_projection(visual_embedding)  # (batch_size, projection_dim)
        audio_proj = self.audio_projection(audio_embedding)      # (batch_size, projection_dim)
        temporal_proj = self.temporal_projection(temporal_embedding)  # (batch_size, projection_dim)
        
        # Stack modalities for attention
        modalities = torch.stack([visual_proj, audio_proj, temporal_proj], dim=1)  # (batch_size, 3, projection_dim)
        
        # Apply self-attention
        attended_modalities, attention_weights = self.attention(modalities, modalities, modalities)
        
        # Add residual connection and layer normalization
        attended_modalities = self.layer_norm(modalities + attended_modalities)
        
        # Global average pooling across modalities
        fused_features = attended_modalities.mean(dim=1)  # (batch_size, projection_dim)
        
        # Apply fusion layers
        logits = self.fusion_layers(fused_features)
        
        return logits


class WeightedFusion(nn.Module):
    """
    Fusion with learnable modality weights
    """
    
    def __init__(self, config):
        super(WeightedFusion, self).__init__()
        
        self.config = config
        self.visual_dim = config.visual.embedding_dim
        self.audio_dim = config.audio.embedding_dim
        self.temporal_dim = config.temporal.embedding_dim
        self.dropout = config.fusion.dropout
        
        # Modality-specific projections
        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        self.audio_projection = nn.Sequential(
            nn.Linear(self.audio_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)  # Equal initial weights
        
        # Fusion layers
        if config.fusion.hidden_dims is None:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = config.fusion.hidden_dims
        
        layers = []
        prev_dim = 256  # After projection
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_layers = nn.Sequential(*layers)
    
    def forward(self, visual_embedding: torch.Tensor, 
                audio_embedding: torch.Tensor, 
                temporal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted fusion
        
        Args:
            visual_embedding: Visual features (batch_size, visual_dim)
            audio_embedding: Audio features (batch_size, audio_dim)
            temporal_embedding: Temporal features (batch_size, temporal_dim)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # Project modalities
        visual_proj = self.visual_projection(visual_embedding)
        audio_proj = self.audio_projection(audio_embedding)
        temporal_proj = self.temporal_projection(temporal_embedding)
        
        # Apply learnable weights
        weights = F.softmax(self.modality_weights, dim=0)
        fused_features = (weights[0] * visual_proj + 
                         weights[1] * audio_proj + 
                         weights[2] * temporal_proj)
        
        # Apply fusion layers
        logits = self.fusion_layers(fused_features)
        
        return logits


class TransformerFusion(nn.Module):
    """
    Fusion using transformer architecture
    """
    
    def __init__(self, config):
        super(TransformerFusion, self).__init__()
        
        self.config = config
        self.visual_dim = config.visual.embedding_dim
        self.audio_dim = config.audio.embedding_dim
        self.temporal_dim = config.temporal.embedding_dim
        self.dropout = config.fusion.dropout
        
        # Project all modalities to same dimension
        self.projection_dim = 256
        self.visual_projection = nn.Linear(self.visual_dim, self.projection_dim)
        self.audio_projection = nn.Linear(self.audio_dim, self.projection_dim)
        self.temporal_projection = nn.Linear(self.temporal_dim, self.projection_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, self.projection_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.projection_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Fusion layers
        if config.fusion.hidden_dims is None:
            hidden_dims = [512, 256, 128]
        else:
            hidden_dims = config.fusion.hidden_dims
        
        layers = []
        prev_dim = self.projection_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_layers = nn.Sequential(*layers)
    
    def forward(self, visual_embedding: torch.Tensor, 
                audio_embedding: torch.Tensor, 
                temporal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with transformer fusion
        
        Args:
            visual_embedding: Visual features (batch_size, visual_dim)
            audio_embedding: Audio features (batch_size, audio_dim)
            temporal_embedding: Temporal features (batch_size, temporal_dim)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        batch_size = visual_embedding.size(0)
        
        # Project all modalities to same dimension
        visual_proj = self.visual_projection(visual_embedding)
        audio_proj = self.audio_projection(audio_embedding)
        temporal_proj = self.temporal_projection(temporal_embedding)
        
        # Stack modalities and add positional encoding
        modalities = torch.stack([visual_proj, audio_proj, temporal_proj], dim=1)
        modalities = modalities + self.pos_encoding
        
        # Apply transformer
        transformed_modalities = self.transformer(modalities)
        
        # Global average pooling
        fused_features = transformed_modalities.mean(dim=1)
        
        # Apply fusion layers
        logits = self.fusion_layers(fused_features)
        
        return logits


def create_fusion_model(config) -> nn.Module:
    """
    Factory function to create fusion model based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Fusion model
    """
    if config.fusion.use_attention:
        return AttentionFusion(config)
    elif hasattr(config.fusion, 'use_transformer') and config.fusion.use_transformer:
        return TransformerFusion(config)
    elif hasattr(config.fusion, 'use_weighted') and config.fusion.use_weighted:
        return WeightedFusion(config)
    else:
        return SimpleFusion(config) 