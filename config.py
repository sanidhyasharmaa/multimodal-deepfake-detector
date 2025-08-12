"""
Configuration file for Multimodal Deepfake Detection System
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class VisualConfig:
    """Configuration for visual stream"""
    model_name: str = "xception"  # or "efficientnetv2_s"
    input_size: Tuple[int, int] = (299, 299)  # Xception default
    embedding_dim: int = 512
    dropout: float = 0.5
    pretrained: bool = True


@dataclass
class AudioConfig:
    """Configuration for audio stream"""
    model_type: str = "wav2vec2"  # or "mfcc_cnn"
    sample_rate: int = 16000
    max_audio_length: float = 10.0  # seconds
    embedding_dim: int = 256
    dropout: float = 0.3
    # For MFCC CNN
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512


@dataclass
class TemporalConfig:
    """Configuration for temporal stream"""
    model_type: str = "gru"  # or "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    embedding_dim: int = 256
    dropout: float = 0.3
    sequence_length: int = 30  # number of frames to process
    use_lip_landmarks: bool = True


@dataclass
class FusionConfig:
    """Configuration for fusion layer"""
    input_dim: int = 1024  # 512 + 256 + 256
    hidden_dims: List[int] = None
    dropout: float = 0.5
    use_attention: bool = True
    attention_heads: int = 8


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"  # or "step", "plateau"
    early_stopping_patience: int = 15
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    
    # Loss function
    loss_type: str = "bce"  # binary cross entropy
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Frame extraction
    frame_interval: int = 5  # extract 1 frame every 5 frames
    max_frames: int = 100
    
    # Video processing
    video_fps: int = 30
    video_size: Tuple[int, int] = (640, 480)
    
    # Audio processing
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    
    # Face detection
    face_detection_method: str = "dlib"  # or "mediapipe"
    face_size: Tuple[int, int] = (224, 224)
    
    # Dataset paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""
    log_dir: str = "logs"
    experiment_name: str = "multimodal_deepfake_detection"
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_interval: int = 100
    save_interval: int = 5  # save checkpoint every N epochs
    
    # Visualization
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_roc_curve: bool = True


@dataclass
class ModelConfig:
    """Main configuration class"""
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    num_workers: int = 4
    
    # Model components
    visual: VisualConfig = None
    audio: AudioConfig = None
    temporal: TemporalConfig = None
    fusion: FusionConfig = None
    
    # Training and data
    training: TrainingConfig = None
    data: DataConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.visual is None:
            self.visual = VisualConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.temporal is None:
            self.temporal = TemporalConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


# Default configuration
config = ModelConfig()

# Model-specific configurations
XCEPTION_CONFIG = ModelConfig(
    visual=VisualConfig(model_name="xception", input_size=(299, 299)),
    audio=AudioConfig(model_type="wav2vec2"),
    temporal=TemporalConfig(model_type="gru", use_lip_landmarks=True),
    fusion=FusionConfig(use_attention=True)
)

EFFICIENTNET_CONFIG = ModelConfig(
    visual=VisualConfig(model_name="efficientnetv2_s", input_size=(384, 384)),
    audio=AudioConfig(model_type="mfcc_cnn"),
    temporal=TemporalConfig(model_type="lstm", use_lip_landmarks=False),
    fusion=FusionConfig(use_attention=False)
) 