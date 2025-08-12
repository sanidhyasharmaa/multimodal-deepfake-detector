"""
Temporal stream model for deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalStream(nn.Module):
    """
    Temporal stream for processing lip landmarks and temporal features
    """
    
    def __init__(self, config):
        super(TemporalStream, self).__init__()
        
        self.config = config
        self.hidden_size = config.temporal.hidden_size
        self.num_layers = config.temporal.num_layers
        self.embedding_dim = config.temporal.embedding_dim
        self.dropout = config.temporal.dropout
        self.sequence_length = config.temporal.sequence_length
        self.use_lip_landmarks = config.temporal.use_lip_landmarks
        
        # Input dimension for lip landmarks (68 points * 2 coordinates)
        self.input_dim = 68 * 2 if self.use_lip_landmarks else 0
        
        if self.input_dim == 0:
            # If not using lip landmarks, create a simple temporal model
            self.temporal_model = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            )
        else:
            # Feature extraction for lip landmarks
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            )
            
            # Temporal modeling with GRU/LSTM
            if config.temporal.model_type == "gru":
                self.temporal_model = nn.GRU(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout if self.num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=True
                )
            elif config.temporal.model_type == "lstm":
                self.temporal_model = nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout if self.num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=True
                )
            else:
                raise ValueError(f"Unsupported temporal model type: {config.temporal.model_type}")
            
            # Final projection
            self.final_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.embedding_dim),  # *2 for bidirectional
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            )
    
    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal stream
        
        Args:
            temporal_data: Temporal data tensor
                - If using lip landmarks: (batch_size, sequence_length, 68, 2)
                - Otherwise: (batch_size, sequence_length, 1)
                
        Returns:
            Temporal embedding tensor of shape (batch_size, embedding_dim)
        """
        if self.input_dim == 0:
            # Simple temporal model without lip landmarks
            batch_size, sequence_length, _ = temporal_data.shape
            
            # Process each timestep
            temporal_features = []
            for i in range(sequence_length):
                features = self.temporal_model(temporal_data[:, i])
                temporal_features.append(features)
            
            # Average across time
            temporal_embedding = torch.stack(temporal_features, dim=1).mean(dim=1)
            
        else:
            # Process lip landmarks
            batch_size, sequence_length, num_points, coords = temporal_data.shape
            
            # Flatten lip landmarks
            temporal_data = temporal_data.view(batch_size, sequence_length, -1)
            
            # Extract features for each timestep
            features = self.feature_extractor(temporal_data)  # (batch_size, sequence_length, hidden_size)
            
            # Apply temporal modeling
            if isinstance(self.temporal_model, nn.GRU):
                temporal_output, hidden = self.temporal_model(features)
            else:  # LSTM
                temporal_output, (hidden, cell) = self.temporal_model(features)
            
            # Use final hidden state (concatenate forward and backward for bidirectional)
            if isinstance(hidden, tuple):
                # For LSTM, hidden is a tuple (hidden, cell)
                final_hidden = hidden[0]
            else:
                # For GRU, hidden is a tensor
                final_hidden = hidden
            
            # Concatenate forward and backward hidden states
            final_hidden = torch.cat([final_hidden[-2], final_hidden[-1]], dim=1)
            
            # Final projection
            temporal_embedding = self.final_projection(final_hidden)
        
        return temporal_embedding


class TemporalStreamWithAttention(nn.Module):
    """
    Temporal stream with self-attention mechanism
    """
    
    def __init__(self, config):
        super(TemporalStreamWithAttention, self).__init__()
        
        self.config = config
        self.hidden_size = config.temporal.hidden_size
        self.embedding_dim = config.temporal.embedding_dim
        self.dropout = config.temporal.dropout
        self.use_lip_landmarks = config.temporal.use_lip_landmarks
        
        # Input dimension for lip landmarks
        self.input_dim = 68 * 2 if self.use_lip_landmarks else 0
        
        if self.input_dim > 0:
            # Feature extraction for lip landmarks
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            temporal_data: Temporal data tensor
            
        Returns:
            Temporal embedding with attention
        """
        if self.input_dim > 0:
            # Process lip landmarks
            batch_size, sequence_length, num_points, coords = temporal_data.shape
            temporal_data = temporal_data.view(batch_size, sequence_length, -1)
        
        # Extract features
        features = self.feature_extractor(temporal_data)  # (batch_size, sequence_length, hidden_size)
        
        # Apply self-attention
        attended_features, _ = self.attention(features, features, features)
        
        # Add residual connection and layer normalization
        attended_features = self.layer_norm(features + attended_features)
        
        # Global average pooling across temporal dimension
        temporal_embedding = attended_features.mean(dim=1)  # (batch_size, hidden_size)
        
        # Final projection
        temporal_embedding = self.final_projection(temporal_embedding)
        
        return temporal_embedding


class OpticalFlowTemporalStream(nn.Module):
    """
    Temporal stream using optical flow for motion analysis
    """
    
    def __init__(self, config):
        super(OpticalFlowTemporalStream, self).__init__()
        
        self.config = config
        self.hidden_size = config.temporal.hidden_size
        self.embedding_dim = config.temporal.embedding_dim
        self.dropout = config.temporal.dropout
        
        # Optical flow feature extraction
        self.flow_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(self.dropout),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(self.dropout),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(self.dropout)
        )
        
        # Temporal modeling
        if config.temporal.model_type == "gru":
            self.temporal_model = nn.GRU(
                input_size=128,
                hidden_size=self.hidden_size,
                num_layers=config.temporal.num_layers,
                dropout=self.dropout if config.temporal.num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.temporal_model = nn.LSTM(
                input_size=128,
                hidden_size=self.hidden_size,
                num_layers=config.temporal.num_layers,
                dropout=self.dropout if config.temporal.num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, optical_flow: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for optical flow
        
        Args:
            optical_flow: Optical flow tensor of shape (batch_size, sequence_length, 2, height, width)
            
        Returns:
            Temporal embedding
        """
        batch_size, sequence_length, channels, height, width = optical_flow.shape
        
        # Process each optical flow frame
        flow_features = []
        for i in range(sequence_length):
            flow = optical_flow[:, i]  # (batch_size, 2, height, width)
            features = self.flow_encoder(flow)  # (batch_size, 128, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (batch_size, 128)
            flow_features.append(features)
        
        # Stack features
        flow_features = torch.stack(flow_features, dim=1)  # (batch_size, sequence_length, 128)
        
        # Apply temporal modeling
        if isinstance(self.temporal_model, nn.GRU):
            temporal_output, hidden = self.temporal_model(flow_features)
        else:
            temporal_output, (hidden, cell) = self.temporal_model(flow_features)
        
        # Use final hidden state
        if isinstance(hidden, tuple):
            final_hidden = hidden[0]
        else:
            final_hidden = hidden
        
        # Concatenate forward and backward hidden states
        final_hidden = torch.cat([final_hidden[-2], final_hidden[-1]], dim=1)
        
        # Final projection
        temporal_embedding = self.final_projection(final_hidden)
        
        return temporal_embedding


def create_temporal_stream(config) -> nn.Module:
    """
    Factory function to create temporal stream based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Temporal stream model
    """
    if hasattr(config.temporal, 'use_optical_flow') and config.temporal.use_optical_flow:
        return OpticalFlowTemporalStream(config)
    elif hasattr(config.temporal, 'use_attention') and config.temporal.use_attention:
        return TemporalStreamWithAttention(config)
    else:
        return TemporalStream(config) 