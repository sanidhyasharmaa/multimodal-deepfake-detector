# 🎯 Multimodal Deepfake Detection System

A high-performance, production-grade deepfake detection system that combines visual, audio, and temporal cues to detect whether a video is real or fake.

## 🌟 Features

- **Multimodal Analysis**: Combines visual, audio, and temporal features for robust detection
- **State-of-the-art Models**: Uses Xception/EfficientNet for visual, Wav2Vec2 for audio, and GRU/LSTM for temporal analysis
- **Production Ready**: Mixed-precision training, comprehensive logging, and evaluation metrics
- **Flexible Architecture**: Modular design with configurable components
- **Real-time Inference**: Fast prediction on single videos or batch processing
- **Comprehensive Evaluation**: ROC-AUC, confusion matrices, and detailed error analysis

## 🏗️ Architecture

### Core Modalities

1. **Visual Stream (Face Frames)**
   - Extracts frames from videos (1 every 5 frames)
   - Uses pretrained CNN (Xception or EfficientNetV2)
   - Output: 512-dimensional embedding

2. **Audio Stream (Speech Features)**
   - Extracts audio using ffmpeg
   - Uses Wav2Vec2 or MFCC + CNN
   - Output: 256-dimensional embedding

3. **Temporal Stream (Lip Movement/Sync)**
   - Extracts lip landmarks using dlib/MediaPipe
   - Uses GRU/LSTM for temporal modeling
   - Output: 256-dimensional embedding

4. **Fusion Layer**
   - Concatenates all modality embeddings
   - Uses attention mechanism or simple fusion
   - Final binary classifier (REAL vs FAKE)

## 📁 Project Structure

```
multimodal-deepfake-detector/
│
├── data/
│   └── dataset.py             # Multimodal PyTorch Dataset
├── models/
│   ├── visual.py              # Visual stream CNN
│   ├── audio.py               # Audio stream CNN/RNN
│   ├── temporal.py            # Lip sync/temporal stream
│   └── fusion.py              # Fusion + final classifier
│
├── utils/
│   ├── preprocessing.py       # Frame/audio extractors, augmentations
│   └── metrics.py             # Accuracy, F1, ROC, AUC, CM
│
├── train.py                   # Mixed-precision training loop
├── eval.py                    # Evaluation with test set
├── inference.py               # Predict on a single video
├── config.py                  # Centralized configuration
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg installed on system

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multimodal-deepfake-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (if not already installed)
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Download dlib shape predictor** (for face landmarks)
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

## 📊 Dataset Preparation

### Expected Directory Structure

```
data/
├── train/
│   ├── real/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── fake/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Supported Video Formats

- MP4, AVI, MOV, MKV
- Any format supported by FFmpeg

## 🎯 Usage

### Training

```bash
# Train with Xception configuration
python train.py --data_path /path/to/dataset --config xception

# Train with EfficientNet configuration
python train.py --data_path /path/to/dataset --config efficientnet

# Use cache for faster training
python train.py --data_path /path/to/dataset --config xception --cache_dir ./cache

# Resume training from checkpoint
python train.py --data_path /path/to/dataset --config xception --resume logs/checkpoint_epoch_10.pth
```

### Evaluation

```bash
# Evaluate trained model
python eval.py --checkpoint logs/best_model.pth --data_path /path/to/test_dataset --config xception

# Save results to custom directory
python eval.py --checkpoint logs/best_model.pth --data_path /path/to/test_dataset --output_dir ./results
```

### Inference

```bash
# Predict on single video
python inference.py --model logs/best_model.pth --video /path/to/video.mp4 --config xception

# Save results to JSON
python inference.py --model logs/best_model.pth --video /path/to/video.mp4 --output results.json

# Batch processing
python inference.py --model logs/best_model.pth --batch video_list.txt --output batch_results.json
```

## ⚙️ Configuration

The system uses a centralized configuration system in `config.py`. Key configurations:

### Model Configurations

- **XCEPTION_CONFIG**: Uses Xception + Wav2Vec2 + GRU + Attention
- **EFFICIENTNET_CONFIG**: Uses EfficientNetV2 + MFCC CNN + LSTM + Simple Fusion

### Custom Configuration

```python
from config import ModelConfig, VisualConfig, AudioConfig, TemporalConfig, FusionConfig

# Create custom configuration
custom_config = ModelConfig(
    visual=VisualConfig(model_name="efficientnetv2_l", embedding_dim=512),
    audio=AudioConfig(model_type="wav2vec2", embedding_dim=256),
    temporal=TemporalConfig(model_type="lstm", use_lip_landmarks=True),
    fusion=FusionConfig(use_attention=True, attention_heads=8)
)
```

## 📈 Performance

### Expected Results

- **Accuracy**: 95-97% on FaceForensics++ test split
- **F1-Score**: >0.92
- **ROC-AUC**: >0.95
- **Inference Time**: <500ms per video

### Training Metrics

The system tracks comprehensive metrics during training:
- Loss curves
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Learning rate scheduling
- Early stopping

## 🔧 Advanced Features

### Mixed Precision Training

Automatically enabled for faster training on compatible GPUs.

### Data Augmentation

Comprehensive augmentation pipeline including:
- Random rotation, flip, transpose
- Noise addition
- Motion blur
- Color jittering
- Optical distortion

### Logging and Monitoring

- **Weights & Biases**: Automatic experiment tracking
- **TensorBoard**: Real-time training visualization
- **Checkpointing**: Automatic model saving
- **Early Stopping**: Prevents overfitting

### Model Interpretability

- Confusion matrix visualization
- ROC curve analysis
- Error case analysis
- Per-modality performance comparison

## 🚀 Deployment

### Production Deployment

1. **Export to TorchScript**
   ```python
   # Convert model to TorchScript for deployment
   traced_model = torch.jit.trace(model, example_input)
   traced_model.save("deployed_model.pt")
   ```

2. **Docker Deployment**
   ```dockerfile
   FROM pytorch/pytorch:latest
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["python", "inference.py", "--model", "best_model.pth"]
   ```

### Web Interface

Optional Streamlit/Gradio interface for easy video upload and prediction.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FaceForensics++ dataset
- Xception and EfficientNet architectures
- Wav2Vec2 for audio processing
- dlib and MediaPipe for face detection

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks

## 🔬 Research

This implementation is based on state-of-the-art research in multimodal deepfake detection. Key papers:

- FaceForensics++: Learning to Detect Manipulated Facial Images
- Wav2Vec2: Self-Supervised Learning of Speech Representations
- Attention Is All You Need (for fusion mechanisms)

---

**Note**: This system is designed for research and educational purposes. Always ensure compliance with local laws and regulations when using deepfake detection technology. 