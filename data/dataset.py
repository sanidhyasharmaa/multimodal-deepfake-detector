"""
Multimodal PyTorch Dataset for Deepfake Detection
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle

from utils.preprocessing import VideoProcessor, FaceDetector, AudioProcessor, DataAugmentation


class MultimodalDeepfakeDataset(Dataset):
    """
    Multimodal dataset for deepfake detection
    
    Loads:
    - Visual frames (face crops)
    - Audio features
    - Temporal features (lip landmarks)
    """
    
    def __init__(self, 
                 data_path: str,
                 config,
                 split: str = "train",
                 max_samples: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize dataset
        
        Args:
            data_path: Path to dataset directory
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            cache_dir: Directory to cache preprocessed data
        """
        self.data_path = data_path
        self.config = config
        self.split = split
        self.cache_dir = cache_dir
        
        # Initialize processors
        self.video_processor = VideoProcessor(
            fps=config.data.video_fps,
            frame_interval=config.data.frame_interval
        )
        
        self.face_detector = FaceDetector(
            method=config.data.face_detection_method,
            face_size=config.visual.input_size
        )
        
        self.audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            max_length=config.audio.max_audio_length
        )
        
        self.augmentation = DataAugmentation(
            input_size=config.visual.input_size,
            prob=config.training.augmentation_prob
        )
        
        # Load data samples
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Create cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _load_samples(self) -> List[Dict]:
        """
        Load dataset samples from directory structure
        
        Expected structure:
        data_path/
        ├── real/
        │   ├── video1.mp4
        │   ├── video2.mp4
        │   └── ...
        └── fake/
            ├── video1.mp4
            ├── video2.mp4
            └── ...
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Load real videos
        real_path = os.path.join(self.data_path, "real")
        if os.path.exists(real_path):
            for video_file in os.listdir(real_path):
                if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(real_path, video_file)
                    samples.append({
                        'video_path': video_path,
                        'label': 0,  # Real
                        'video_id': video_file.split('.')[0]
                    })
        
        # Load fake videos
        fake_path = os.path.join(self.data_path, "fake")
        if os.path.exists(fake_path):
            for video_file in os.listdir(fake_path):
                if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(fake_path, video_file)
                    samples.append({
                        'video_path': video_path,
                        'label': 1,  # Fake
                        'video_id': video_file.split('.')[0]
                    })
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        print(f"Real videos: {sum(1 for s in samples if s['label'] == 0)}")
        print(f"Fake videos: {sum(1 for s in samples if s['label'] == 1)}")
        
        return samples
    
    def _get_cache_path(self, video_id: str, modality: str) -> str:
        """Get cache file path for a specific modality"""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{video_id}_{modality}.pkl")
    
    def _load_from_cache(self, cache_path: str) -> Optional[object]:
        """Load data from cache"""
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, data: object, cache_path: str):
        """Save data to cache"""
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except:
                pass
    
    def _extract_visual_features(self, video_path: str, video_id: str) -> torch.Tensor:
        """Extract visual features from video frames"""
        cache_path = self._get_cache_path(video_id, "visual")
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data is not None:
            return cached_data
        
        # Extract frames
        frames = self.video_processor.extract_frames(
            video_path, 
            max_frames=self.config.data.max_frames
        )
        
        if len(frames) == 0:
            # Return zero tensor if no frames extracted
            return torch.zeros(self.config.data.max_frames, 3, *self.config.visual.input_size)
        
        # Detect faces and apply augmentation
        processed_frames = []
        is_training = self.split == "train"
        
        for frame in frames:
            # Detect face
            face_crop = self.face_detector.detect_face(frame)
            
            if face_crop is None:
                # If no face detected, use the original frame
                face_crop = cv2.resize(frame, self.config.visual.input_size)
            
            # Apply augmentation
            processed_frame = self.augmentation(face_crop, is_training=is_training)
            processed_frames.append(processed_frame)
        
        # Pad or truncate to max_frames
        while len(processed_frames) < self.config.data.max_frames:
            processed_frames.append(torch.zeros(3, *self.config.visual.input_size))
        
        processed_frames = processed_frames[:self.config.data.max_frames]
        visual_features = torch.stack(processed_frames)
        
        # Cache the result
        self._save_to_cache(visual_features, cache_path)
        
        return visual_features
    
    def _extract_audio_features(self, video_path: str, video_id: str) -> torch.Tensor:
        """Extract audio features from video"""
        cache_path = self._get_cache_path(video_id, "audio")
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Extract audio from video
            audio_path = self.video_processor.extract_audio(video_path)
            
            # Load and process audio
            audio = self.audio_processor.load_audio(audio_path)
            
            if self.config.audio.model_type == "mfcc_cnn":
                # Extract MFCC features
                mfcc = self.audio_processor.extract_mfcc(
                    audio, 
                    n_mfcc=self.config.audio.n_mfcc
                )
                audio_features = torch.from_numpy(mfcc).float()
            else:
                # For wav2vec2, return raw audio
                audio_features = torch.from_numpy(audio).float()
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Cache the result
            self._save_to_cache(audio_features, cache_path)
            
            return audio_features
            
        except Exception as e:
            print(f"Error processing audio for {video_path}: {e}")
            # Return zero tensor on error
            if self.config.audio.model_type == "mfcc_cnn":
                return torch.zeros(self.config.audio.n_mfcc, 100)  # Default MFCC size
            else:
                return torch.zeros(self.config.audio.sample_rate * int(self.config.audio.max_audio_length))
    
    def _extract_temporal_features(self, video_path: str, video_id: str) -> torch.Tensor:
        """Extract temporal features (lip landmarks) from video"""
        if not self.config.temporal.use_lip_landmarks:
            # Return zero tensor if lip landmarks not used
            return torch.zeros(self.config.temporal.sequence_length, 68, 2)
        
        cache_path = self._get_cache_path(video_id, "temporal")
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Extract frames for temporal analysis
            frames = self.video_processor.extract_frames(
                video_path, 
                max_frames=self.config.temporal.sequence_length
            )
            
            lip_landmarks = []
            
            for frame in frames:
                landmarks = self.face_detector.extract_lip_landmarks(frame)
                
                if landmarks is not None:
                    lip_landmarks.append(landmarks)
                else:
                    # Use zero landmarks if detection fails
                    lip_landmarks.append(np.zeros((68, 2)))
            
            # Pad or truncate to sequence_length
            while len(lip_landmarks) < self.config.temporal.sequence_length:
                lip_landmarks.append(np.zeros((68, 2)))
            
            lip_landmarks = lip_landmarks[:self.config.temporal.sequence_length]
            temporal_features = torch.from_numpy(np.array(lip_landmarks)).float()
            
            # Cache the result
            self._save_to_cache(temporal_features, cache_path)
            
            return temporal_features
            
        except Exception as e:
            print(f"Error processing temporal features for {video_path}: {e}")
            return torch.zeros(self.config.temporal.sequence_length, 68, 2)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
            - visual: Visual features (frames)
            - audio: Audio features
            - temporal: Temporal features (lip landmarks)
            - label: Ground truth label
            - video_id: Video identifier
        """
        sample = self.samples[idx]
        video_path = sample['video_path']
        video_id = sample['video_id']
        label = sample['label']
        
        # Extract features for all modalities
        visual_features = self._extract_visual_features(video_path, video_id)
        audio_features = self._extract_audio_features(video_path, video_id)
        temporal_features = self._extract_temporal_features(video_path, video_id)
        
        return {
            'visual': visual_features,
            'audio': audio_features,
            'temporal': temporal_features,
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id
        }


class MultimodalDataLoader:
    """Custom data loader for multimodal data"""
    
    def __init__(self, dataset: MultimodalDeepfakeDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        """Create batches of multimodal data"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Collate batch data
            batch = {
                'visual': torch.stack([item['visual'] for item in batch_data]),
                'audio': [item['audio'] for item in batch_data],  # Variable length
                'temporal': torch.stack([item['temporal'] for item in batch_data]),
                'label': torch.stack([item['label'] for item in batch_data]),
                'video_id': [item['video_id'] for item in batch_data]
            }
            
            yield batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_data_loaders(config, data_path: str, cache_dir: Optional[str] = None):
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration object
        data_path: Path to dataset directory
        cache_dir: Directory to cache preprocessed data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MultimodalDeepfakeDataset(
        data_path=os.path.join(data_path, "train"),
        config=config,
        split="train",
        cache_dir=os.path.join(cache_dir, "train") if cache_dir else None
    )
    
    val_dataset = MultimodalDeepfakeDataset(
        data_path=os.path.join(data_path, "val"),
        config=config,
        split="val",
        cache_dir=os.path.join(cache_dir, "val") if cache_dir else None
    )
    
    test_dataset = MultimodalDeepfakeDataset(
        data_path=os.path.join(data_path, "test"),
        config=config,
        split="test",
        cache_dir=os.path.join(cache_dir, "test") if cache_dir else None
    )
    
    # Create data loaders
    train_loader = MultimodalDataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True
    )
    
    val_loader = MultimodalDataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False
    )
    
    test_loader = MultimodalDataLoader(
        test_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader 