"""
Visual stream model for deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple


class VisualStream(nn.Module):
    """
    Visual stream for processing face frames
    
    Uses pretrained CNN (Xception or EfficientNet) to extract features from face frames
    """
    
    def __init__(self, config):
        """
        Initialize visual stream
        
        Args:
            config: Configuration object containing visual settings
        """
        super(VisualStream, self).__init__()
        
        self.config = config
        self.model_name = config.visual.model_name
        self.embedding_dim = config.visual.embedding_dim
        self.dropout = config.visual.dropout
        self.pretrained = config.visual.pretrained
        
        # Load pretrained model
        self.backbone = self._load_backbone()
        
        # Get feature dimension from backbone
        if hasattr(self.backbone, 'num_features'):
            backbone_features = self.backbone.num_features
        elif hasattr(self.backbone, 'classifier'):
            # For models like Xception
            backbone_features = self.backbone.classifier.in_features
        else:
            # Default fallback
            backbone_features = 2048
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(backbone_features, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        # Frame aggregation (temporal pooling)
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
    
    def _load_backbone(self) -> nn.Module:
        """Load pretrained backbone model"""
        if self.model_name == "xception":
            model = timm.create_model(
                'xception',
                pretrained=self.pretrained,
                num_classes=0,  # Remove classifier
                global_pool=''
            )
        elif self.model_name == "efficientnetv2_s":
            model = timm.create_model(
                'efficientnetv2_s',
                pretrained=self.pretrained,
                num_classes=0,  # Remove classifier
                global_pool=''
            )
        elif self.model_name == "efficientnetv2_m":
            model = timm.create_model(
                'efficientnetv2_m',
                pretrained=self.pretrained,
                num_classes=0,
                global_pool=''
            )
        elif self.model_name == "efficientnetv2_l":
            model = timm.create_model(
                'efficientnetv2_l',
                pretrained=self.pretrained,
                num_classes=0,
                global_pool=''
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through visual stream
        
        Args:
            frames: Input frames tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Visual embedding tensor of shape (batch_size, embedding_dim)
        """
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Reshape frames for batch processing
        frames_flat = frames.view(batch_size * num_frames, channels, height, width)
        
        # Extract features from backbone
        features = self.backbone(frames_flat)  # (batch_size * num_frames, backbone_features, H, W)
        
        # Apply feature extraction
        features = self.feature_extractor(features)  # (batch_size * num_frames, embedding_dim)
        
        # Reshape back to separate frames
        features = features.view(batch_size, num_frames, self.embedding_dim)
        
        # Temporal pooling across frames
        features = features.transpose(1, 2)  # (batch_size, embedding_dim, num_frames)
        features = self.temporal_pooling(features)  # (batch_size, embedding_dim, 1)
        features = features.squeeze(-1)  # (batch_size, embedding_dim)
        
        # Final embedding projection
        visual_embedding = self.embedding_projection(features)
        
        return visual_embedding


class VisualStreamWithAttention(nn.Module):
    """
    Visual stream with self-attention mechanism for better temporal modeling
    """
    
    def __init__(self, config):
        super(VisualStreamWithAttention, self).__init__()
        
        self.config = config
        self.embedding_dim = config.visual.embedding_dim
        
        # Base visual stream
        self.visual_stream = VisualStream(config)
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.visual.dropout)
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            frames: Input frames tensor
            
        Returns:
            Visual embedding with attention
        """
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Extract features for each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i]  # (batch_size, channels, height, width)
            feature = self.visual_stream.backbone(frame)
            feature = self.visual_stream.feature_extractor(feature)
            frame_features.append(feature)
        
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)  # (batch_size, num_frames, embedding_dim)
        
        # Apply self-attention
        attended_features, _ = self.attention(frame_features, frame_features, frame_features)
        
        # Add residual connection and layer normalization
        attended_features = self.layer_norm(frame_features + attended_features)
        
        # Global average pooling across temporal dimension
        visual_embedding = attended_features.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Final projection
        visual_embedding = self.final_projection(visual_embedding)
        
        return visual_embedding


class VisualStreamWithLSTM(nn.Module):
    """
    Visual stream with LSTM for temporal modeling
    """
    
    def __init__(self, config):
        super(VisualStreamWithLSTM, self).__init__()
        
        self.config = config
        self.embedding_dim = config.visual.embedding_dim
        
        # Base visual stream (without temporal pooling)
        self.visual_stream = VisualStream(config)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(config.visual.dropout)
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LSTM
        
        Args:
            frames: Input frames tensor
            
        Returns:
            Visual embedding with LSTM
        """
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Extract features for each frame
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i]
            feature = self.visual_stream.backbone(frame)
            feature = self.visual_stream.feature_extractor(feature)
            frame_features.append(feature)
        
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)  # (batch_size, num_frames, embedding_dim)
        
        # Apply LSTM
        lstm_output, (hidden, cell) = self.lstm(frame_features)
        
        # Use final hidden state (concatenate forward and backward)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, embedding_dim * 2)
        
        # Final projection
        visual_embedding = self.final_projection(final_hidden)
        
        return visual_embedding


def create_visual_stream(config) -> nn.Module:
    """
    Factory function to create visual stream based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Visual stream model
    """
    if hasattr(config.visual, 'use_attention') and config.visual.use_attention:
        return VisualStreamWithAttention(config)
    elif hasattr(config.visual, 'use_lstm') and config.visual.use_lstm:
        return VisualStreamWithLSTM(config)
    else:
        return VisualStream(config) 