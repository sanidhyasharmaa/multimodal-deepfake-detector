"""
Inference script for multimodal deepfake detection on single videos
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, XCEPTION_CONFIG, EFFICIENTNET_CONFIG
from models.visual import create_visual_stream
from models.audio import create_audio_stream
from models.temporal import create_temporal_stream
from models.fusion import create_fusion_model
from utils.preprocessing import VideoProcessor, FaceDetector, AudioProcessor


class MultimodalDeepfakeDetector(nn.Module):
    """
    Complete multimodal deepfake detection model
    """
    
    def __init__(self, config):
        super(MultimodalDeepfakeDetector, self).__init__()
        
        self.config = config
        
        # Initialize modality streams
        self.visual_stream = create_visual_stream(config)
        self.audio_stream = create_audio_stream(config)
        self.temporal_stream = create_temporal_stream(config)
        
        # Initialize fusion model
        self.fusion_model = create_fusion_model(config)
    
    def forward(self, visual_data: torch.Tensor, 
                audio_data: torch.Tensor, 
                temporal_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model
        
        Args:
            visual_data: Visual frames (batch_size, num_frames, channels, height, width)
            audio_data: Audio features (batch_size, audio_length)
            temporal_data: Temporal features (batch_size, sequence_length, num_points, coords)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # Process each modality
        visual_embedding = self.visual_stream(visual_data)
        audio_embedding = self.audio_stream(audio_data)
        temporal_embedding = self.temporal_stream(temporal_data)
        
        # Fuse modalities
        logits = self.fusion_model(visual_embedding, audio_embedding, temporal_embedding)
        
        return logits


class VideoInference:
    """
    Video inference class for real-time deepfake detection
    """
    
    def __init__(self, config, model_path: str):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
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
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        print(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = MultimodalDeepfakeDetector(self.config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def preprocess_video(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess video for inference
        
        Args:
            video_path: Path to input video
            
        Returns:
            Tuple of (visual_features, audio_features, temporal_features)
        """
        print("Preprocessing video...")
        
        # Extract frames
        frames = self.video_processor.extract_frames(
            video_path, 
            max_frames=self.config.data.max_frames
        )
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        # Process visual features
        visual_features = self.process_visual_features(frames)
        
        # Process audio features
        audio_features = self.process_audio_features(video_path)
        
        # Process temporal features
        temporal_features = self.process_temporal_features(frames)
        
        return visual_features, audio_features, temporal_features
    
    def process_visual_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Process visual frames"""
        processed_frames = []
        
        for frame in frames:
            # Detect face
            face_crop = self.face_detector.detect_face(frame)
            
            if face_crop is None:
                # If no face detected, use the original frame
                face_crop = cv2.resize(frame, self.config.visual.input_size)
            
            # Convert to tensor and normalize
            face_tensor = torch.from_numpy(face_crop).float()
            face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
            face_tensor = face_tensor / 255.0  # Normalize to [0, 1]
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            face_tensor = (face_tensor - mean) / std
            
            processed_frames.append(face_tensor)
        
        # Pad or truncate to max_frames
        while len(processed_frames) < self.config.data.max_frames:
            processed_frames.append(torch.zeros(3, *self.config.visual.input_size))
        
        processed_frames = processed_frames[:self.config.data.max_frames]
        visual_features = torch.stack(processed_frames)
        
        return visual_features
    
    def process_audio_features(self, video_path: str) -> torch.Tensor:
        """Process audio features"""
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
            
            return audio_features
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            # Return zero tensor on error
            if self.config.audio.model_type == "mfcc_cnn":
                return torch.zeros(self.config.audio.n_mfcc, 100)
            else:
                return torch.zeros(self.config.audio.sample_rate * int(self.config.audio.max_audio_length))
    
    def process_temporal_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Process temporal features (lip landmarks)"""
        if not self.config.temporal.use_lip_landmarks:
            # Return zero tensor if lip landmarks not used
            return torch.zeros(self.config.temporal.sequence_length, 68, 2)
        
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
        
        return temporal_features
    
    def predict(self, video_path: str) -> Dict[str, float]:
        """
        Predict deepfake probability for a video
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess video
        visual_features, audio_features, temporal_features = self.preprocess_video(video_path)
        
        # Add batch dimension
        visual_features = visual_features.unsqueeze(0).to(self.device)
        audio_features = audio_features.unsqueeze(0).to(self.device)
        temporal_features = temporal_features.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(visual_features, audio_features, temporal_features)
            probability = torch.sigmoid(logits.squeeze()).item()
        
        # Determine prediction
        prediction = "FAKE" if probability > 0.5 else "REAL"
        confidence = max(probability, 1 - probability)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)
        
        results = {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'processing_time': processing_time,
            'video_info': video_info,
            'model_config': {
                'visual_model': self.config.visual.model_name,
                'audio_model': self.config.audio.model_type,
                'temporal_model': self.config.temporal.model_type,
                'fusion_method': 'attention' if self.config.fusion.use_attention else 'simple'
            }
        }
        
        return results
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict[str, float]]:
        """
        Predict deepfake probability for multiple videos
        
        Args:
            video_paths: List of video paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    'prediction': 'ERROR',
                    'probability': 0.0,
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'error': str(e)
                })
        
        return results


def print_results(results: Dict[str, float]):
    """Print prediction results in a formatted way"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION RESULTS")
    print("="*60)
    
    print(f"Prediction: {results['prediction']}")
    print(f"Probability: {results['probability']:.4f}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    
    if 'video_info' in results:
        info = results['video_info']
        print(f"\nVideo Information:")
        print(f"  Duration: {info['duration']:.2f} seconds")
        print(f"  FPS: {info['fps']:.2f}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  Frame Count: {info['frame_count']}")
    
    if 'model_config' in results:
        config = results['model_config']
        print(f"\nModel Configuration:")
        print(f"  Visual: {config['visual_model']}")
        print(f"  Audio: {config['audio_model']}")
        print(f"  Temporal: {config['temporal_model']}")
        print(f"  Fusion: {config['fusion_method']}")
    
    print("="*60)


def save_results(results: Dict[str, float], output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Inference for multimodal deepfake detection")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="xception", 
                       choices=["xception", "efficientnet"], help="Model configuration")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--batch", type=str, default=None, help="File containing list of video paths for batch processing")
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == "xception":
        model_config = XCEPTION_CONFIG
    else:
        model_config = EFFICIENTNET_CONFIG
    
    # Create inference object
    inference = VideoInference(model_config, args.model)
    
    if args.batch:
        # Batch processing
        with open(args.batch, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(video_paths)} videos...")
        results = inference.predict_batch(video_paths)
        
        # Save batch results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Batch results saved to: {args.output}")
        
        # Print summary
        predictions = [r['prediction'] for r in results if r['prediction'] != 'ERROR']
        fake_count = sum(1 for p in predictions if p == 'FAKE')
        real_count = sum(1 for p in predictions if p == 'REAL')
        
        print(f"\nBatch Processing Summary:")
        print(f"Total videos: {len(video_paths)}")
        print(f"Successful predictions: {len(predictions)}")
        print(f"Real videos: {real_count}")
        print(f"Fake videos: {fake_count}")
        print(f"Errors: {len(results) - len(predictions)}")
        
    else:
        # Single video processing
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        
        results = inference.predict(args.video)
        
        # Print results
        print_results(results)
        
        # Save results
        if args.output:
            save_results(results, args.output)


if __name__ == "__main__":
    main() 