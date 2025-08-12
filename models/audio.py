"""
Audio stream model for deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Tuple


class MFCCCNN(nn.Module):
    """
    CNN-based model for MFCC features
    """
    
    def __init__(self, config):
        super(MFCCCNN, self).__init__()
        
        self.config = config
        self.n_mfcc = config.audio.n_mfcc
        self.embedding_dim = config.audio.embedding_dim
        self.dropout = config.audio.dropout
        
        # CNN layers for MFCC processing
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
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
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(self.dropout),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(self.dropout)
        )
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MFCC features
        
        Args:
            mfcc: MFCC features of shape (batch_size, n_mfcc, time_steps)
            
        Returns:
            Audio embedding of shape (batch_size, embedding_dim)
        """
        # Add channel dimension
        x = mfcc.unsqueeze(1)  # (batch_size, 1, n_mfcc, time_steps)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Apply projection
        audio_embedding = self.projection(x)
        
        return audio_embedding


class Wav2Vec2AudioStream(nn.Module):
    """
    Audio stream using Wav2Vec2 model
    """
    
    def __init__(self, config):
        super(Wav2Vec2AudioStream, self).__init__()
        
        self.config = config
        self.embedding_dim = config.audio.embedding_dim
        self.dropout = config.audio.dropout
        
        # Load Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze Wav2Vec2 parameters (optional)
        if hasattr(config.audio, 'freeze_wav2vec2') and config.audio.freeze_wav2vec2:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Get Wav2Vec2 output dimension
        wav2vec2_dim = self.wav2vec2.config.hidden_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(wav2vec2_dim, wav2vec2_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(wav2vec2_dim // 2, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        # Temporal pooling
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio features
        
        Args:
            audio: Audio tensor of shape (batch_size, audio_length)
            
        Returns:
            Audio embedding of shape (batch_size, embedding_dim)
        """
        # Wav2Vec2 expects input in range [-1, 1]
        if audio.max() > 1.0:
            audio = audio / 32768.0  # Normalize from int16 range
        
        # Get Wav2Vec2 features
        with torch.no_grad() if not self.wav2vec2.training else torch.enable_grad():
            wav2vec2_output = self.wav2vec2(audio)
            features = wav2vec2_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Apply feature extraction
        features = self.feature_extractor(features)  # (batch_size, seq_len, embedding_dim)
        
        # Temporal pooling
        features = features.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        features = self.temporal_pooling(features)  # (batch_size, embedding_dim, 1)
        features = features.squeeze(-1)  # (batch_size, embedding_dim)
        
        # Final projection
        audio_embedding = self.final_projection(features)
        
        return audio_embedding


class AudioStreamWithLSTM(nn.Module):
    """
    Audio stream with LSTM for temporal modeling
    """
    
    def __init__(self, config):
        super(AudioStreamWithLSTM, self).__init__()
        
        self.config = config
        self.embedding_dim = config.audio.embedding_dim
        self.dropout = config.audio.dropout
        
        # Base audio stream
        if config.audio.model_type == "wav2vec2":
            self.audio_stream = Wav2Vec2AudioStream(config)
        else:
            self.audio_stream = MFCCCNN(config)
        
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
            nn.Dropout(self.dropout)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LSTM
        
        Args:
            audio: Audio tensor
            
        Returns:
            Audio embedding with LSTM
        """
        # Get base audio features
        audio_features = self.audio_stream(audio)
        
        # Reshape for LSTM (assuming we have temporal information)
        # For simplicity, we'll treat the embedding as a single timestep
        audio_features = audio_features.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Apply LSTM
        lstm_output, (hidden, cell) = self.lstm(audio_features)
        
        # Use final hidden state
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, embedding_dim * 2)
        
        # Final projection
        audio_embedding = self.final_projection(final_hidden)
        
        return audio_embedding


class AudioStreamWithAttention(nn.Module):
    """
    Audio stream with self-attention mechanism
    """
    
    def __init__(self, config):
        super(AudioStreamWithAttention, self).__init__()
        
        self.config = config
        self.embedding_dim = config.audio.embedding_dim
        
        # Base audio stream
        if config.audio.model_type == "wav2vec2":
            self.audio_stream = Wav2Vec2AudioStream(config)
        else:
            self.audio_stream = MFCCCNN(config)
        
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
            nn.Dropout(config.audio.dropout)
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            audio: Audio tensor
            
        Returns:
            Audio embedding with attention
        """
        # Get base audio features
        audio_features = self.audio_stream(audio)
        
        # Reshape for attention (treat as single timestep)
        audio_features = audio_features.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Apply self-attention
        attended_features, _ = self.attention(audio_features, audio_features, audio_features)
        
        # Add residual connection and layer normalization
        attended_features = self.layer_norm(audio_features + attended_features)
        
        # Global average pooling
        audio_embedding = attended_features.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Final projection
        audio_embedding = self.final_projection(audio_embedding)
        
        return audio_embedding


def create_audio_stream(config) -> nn.Module:
    """
    Factory function to create audio stream based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Audio stream model
    """
    if config.audio.model_type == "wav2vec2":
        if hasattr(config.audio, 'use_attention') and config.audio.use_attention:
            return AudioStreamWithAttention(config)
        elif hasattr(config.audio, 'use_lstm') and config.audio.use_lstm:
            return AudioStreamWithLSTM(config)
        else:
            return Wav2Vec2AudioStream(config)
    elif config.audio.model_type == "mfcc_cnn":
        if hasattr(config.audio, 'use_attention') and config.audio.use_attention:
            return AudioStreamWithAttention(config)
        elif hasattr(config.audio, 'use_lstm') and config.audio.use_lstm:
            return AudioStreamWithLSTM(config)
        else:
            return MFCCCNN(config)
    else:
        raise ValueError(f"Unsupported audio model type: {config.audio.model_type}") 