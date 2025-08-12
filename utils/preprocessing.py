"""
Preprocessing utilities for multimodal deepfake detection
"""
import os
import cv2
import numpy as np
import torch
import torchaudio
import librosa
import ffmpeg
from typing import List, Tuple, Optional, Union
import dlib
import mediapipe as mp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VideoProcessor:
    """Video processing utilities"""
    
    def __init__(self, fps: int = 30, frame_interval: int = 5):
        self.fps = fps
        self.frame_interval = frame_interval
    
    def extract_frames(self, video_path: str, max_frames: int = 100) -> List[np.ndarray]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_interval == 0 and extracted_count < max_frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        return frames
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video using ffmpeg
        
        Args:
            video_path: Path to video file
            output_path: Path to save extracted audio (optional)
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = video_path.replace('.mp4', '_audio.wav')
        
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16000')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg error: {e}")
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info


class FaceDetector:
    """Face detection and landmark extraction"""
    
    def __init__(self, method: str = "dlib", face_size: Tuple[int, int] = (224, 224)):
        self.method = method
        self.face_size = face_size
        
        if method == "dlib":
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        elif method == "mediapipe":
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        else:
            raise ValueError(f"Unsupported face detection method: {method}")
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop face from image
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Cropped face image or None if no face detected
        """
        if self.method == "dlib":
            return self._detect_face_dlib(image)
        elif self.method == "mediapipe":
            return self._detect_face_mediapipe(image)
    
    def _detect_face_dlib(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face using dlib"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        face = max(faces, key=lambda rect: rect.area())
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        # Crop and resize face
        face_crop = image[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, self.face_size)
        
        return face_crop
    
    def _detect_face_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face using MediaPipe"""
        results = self.mp_face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get face landmarks
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Calculate bounding box from landmarks
        x_coords = [int(landmark.x * w) for landmark in landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in landmarks.landmark]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Add padding
        padding = int((x2 - x1) * 0.1)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop and resize face
        face_crop = image[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, self.face_size)
        
        return face_crop
    
    def extract_lip_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract lip landmarks for temporal analysis
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Lip landmarks as numpy array or None if no face detected
        """
        if self.method == "dlib":
            return self._extract_lip_landmarks_dlib(image)
        elif self.method == "mediapipe":
            return self._extract_lip_landmarks_mediapipe(image)
    
    def _extract_lip_landmarks_dlib(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract lip landmarks using dlib"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Lip landmarks (indices 48-68)
        lip_points = []
        for i in range(48, 68):
            point = landmarks.part(i)
            lip_points.append([point.x, point.y])
        
        return np.array(lip_points)
    
    def _extract_lip_landmarks_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract lip landmarks using MediaPipe"""
        results = self.mp_face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Lip landmarks (indices 61-84 for outer lip, 85-96 for inner lip)
        lip_indices = list(range(61, 85)) + list(range(85, 97))
        lip_points = []
        
        for idx in lip_indices:
            landmark = landmarks.landmark[idx]
            lip_points.append([int(landmark.x * w), int(landmark.y * h)])
        
        return np.array(lip_points)


class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self, sample_rate: int = 16000, max_length: float = 10.0):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio as numpy array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pad or truncate to max_length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), 'constant')
        
        return audio
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features from audio
        
        Args:
            audio: Input audio as numpy array
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features as numpy array
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        return mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Extract mel spectrogram from audio
        
        Args:
            audio: Input audio as numpy array
            n_mels: Number of mel bands
            
        Returns:
            Mel spectrogram as numpy array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec


class DataAugmentation:
    """Data augmentation for training"""
    
    def __init__(self, input_size: Tuple[int, int] = (299, 299), prob: float = 0.5):
        self.input_size = input_size
        self.prob = prob
        
        # Visual augmentations
        self.visual_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(*self.input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Test transform (no augmentation)
        self.test_transform = A.Compose([
            A.Resize(*self.input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __call__(self, image: np.ndarray, is_training: bool = True) -> torch.Tensor:
        """
        Apply augmentation to image
        
        Args:
            image: Input image as numpy array (RGB)
            is_training: Whether to apply training augmentations
            
        Returns:
            Augmented image as torch tensor
        """
        if is_training and np.random.random() < self.prob:
            augmented = self.visual_transform(image=image)
        else:
            augmented = self.test_transform(image=image)
        
        return augmented['image'] 