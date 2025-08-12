"""
Training script for multimodal deepfake detection
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, XCEPTION_CONFIG, EFFICIENTNET_CONFIG
from data.dataset import create_data_loaders
from models.visual import create_visual_stream
from models.audio import create_audio_stream
from models.temporal import create_temporal_stream
from models.fusion import create_fusion_model
from utils.metrics import DeepfakeMetrics, MetricsVisualizer, print_metrics_summary


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
            audio_data: Audio features (batch_size, audio_length) or list of variable length
            temporal_data: Temporal features (batch_size, sequence_length, num_points, coords)
            
        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # Process each modality
        visual_embedding = self.visual_stream(visual_data)
        
        # Handle variable length audio
        if isinstance(audio_data, list):
            audio_embeddings = []
            for audio in audio_data:
                audio_embedding = self.audio_stream(audio.unsqueeze(0))
                audio_embeddings.append(audio_embedding)
            audio_embedding = torch.cat(audio_embeddings, dim=0)
        else:
            audio_embedding = self.audio_stream(audio_data)
        
        temporal_embedding = self.temporal_stream(temporal_data)
        
        # Fuse modalities
        logits = self.fusion_model(visual_embedding, audio_embedding, temporal_embedding)
        
        return logits


class Trainer:
    """
    Training class for multimodal deepfake detection
    """
    
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize scheduler
        if config.training.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        elif config.training.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif config.training.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        
        # Initialize loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Initialize metrics
        self.train_metrics = DeepfakeMetrics()
        self.val_metrics = DeepfakeMetrics()
        
        # Initialize logging
        self.setup_logging()
        
        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1s = []
        self.val_f1s = []
        self.train_roc_aucs = []
        self.val_roc_aucs = []
    
    def setup_logging(self):
        """Setup logging with wandb and tensorboard"""
        # Create log directory
        os.makedirs(self.config.logging.log_dir, exist_ok=True)
        
        # Setup wandb
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.experiment_name,
                config=vars(self.config),
                name=f"{self.config.logging.experiment_name}_{int(time.time())}"
            )
        
        # Setup tensorboard
        if self.config.logging.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(self.config.logging.log_dir, "tensorboard")
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} - Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            visual_data = batch['visual'].to(self.device)
            audio_data = batch['audio']
            temporal_data = batch['temporal'].to(self.device)
            labels = batch['label'].float().to(self.device)
            
            # Handle variable length audio
            if isinstance(audio_data, list):
                audio_data = [audio.to(self.device) for audio in audio_data]
            else:
                audio_data = audio_data.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.training.mixed_precision and self.scaler is not None:
                with autocast():
                    logits = self.model(visual_data, audio_data, temporal_data)
                    loss = self.criterion(logits.squeeze(), labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits = self.model(visual_data, audio_data, temporal_data)
                loss = self.criterion(logits.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update metrics
            predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            probabilities = torch.sigmoid(logits.squeeze())
            
            self.train_metrics.update(predictions, labels, probabilities)
            
            # Update loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{self.train_metrics.compute().get('accuracy', 0):.4f}"
            })
            
            # Log to wandb
            if self.config.logging.use_wandb and batch_idx % self.config.logging.log_interval == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_accuracy': self.train_metrics.compute().get('accuracy', 0)
                })
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute()
        epoch_loss = total_loss / num_batches
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} - Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                visual_data = batch['visual'].to(self.device)
                audio_data = batch['audio']
                temporal_data = batch['temporal'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                # Handle variable length audio
                if isinstance(audio_data, list):
                    audio_data = [audio.to(self.device) for audio in audio_data]
                else:
                    audio_data = audio_data.to(self.device)
                
                # Forward pass
                logits = self.model(visual_data, audio_data, temporal_data)
                loss = self.criterion(logits.squeeze(), labels)
                
                # Update metrics
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                probabilities = torch.sigmoid(logits.squeeze())
                
                self.val_metrics.update(predictions, labels, probabilities)
                
                # Update loss
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{self.val_metrics.compute().get('accuracy', 0):.4f}"
                })
        
        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute()
        epoch_loss = total_loss / num_batches
        
        return {
            'loss': epoch_loss,
            **epoch_metrics
        }
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.training.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.config.training.scheduler_type == "plateau":
                self.scheduler.step(val_metrics['f1'])
            else:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.train_f1s.append(train_metrics['f1'])
            self.val_f1s.append(val_metrics['f1'])
            
            if 'roc_auc' in train_metrics:
                self.train_roc_aucs.append(train_metrics['roc_auc'])
            if 'roc_auc' in val_metrics:
                self.val_roc_aucs.append(val_metrics['roc_auc'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print_metrics_summary(train_metrics, "Training Metrics")
            print_metrics_summary(val_metrics, "Validation Metrics")
            
            # Log to wandb
            if self.config.logging.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Add training metrics
                for key, value in train_metrics.items():
                    log_dict[f'train/{key}'] = value
                
                # Add validation metrics
                for key, value in val_metrics.items():
                    log_dict[f'val/{key}'] = value
                
                wandb.log(log_dict)
            
            # Log to tensorboard
            if self.config.logging.use_tensorboard:
                for key, value in train_metrics.items():
                    self.tensorboard_writer.add_scalar(f'train/{key}', value, epoch)
                
                for key, value in val_metrics.items():
                    self.tensorboard_writer.add_scalar(f'val/{key}', value, epoch)
                
                self.tensorboard_writer.add_scalar('learning_rate', 
                                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Final evaluation
        print("\nTraining completed!")
        self.evaluate()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.logging.log_dir, 
            f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.logging.log_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with F1: {metrics['f1']:.4f}")
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(self.config.logging.log_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Evaluate
        self.model.eval()
        test_metrics = DeepfakeMetrics()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                visual_data = batch['visual'].to(self.device)
                audio_data = batch['audio']
                temporal_data = batch['temporal'].to(self.device)
                labels = batch['label'].float().to(self.device)
                
                # Handle variable length audio
                if isinstance(audio_data, list):
                    audio_data = [audio.to(self.device) for audio in audio_data]
                else:
                    audio_data = audio_data.to(self.device)
                
                # Forward pass
                logits = self.model(visual_data, audio_data, temporal_data)
                
                # Update metrics
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                probabilities = torch.sigmoid(logits.squeeze())
                
                test_metrics.update(predictions, labels, probabilities)
        
        # Compute final metrics
        final_metrics = test_metrics.compute()
        print_metrics_summary(final_metrics, "Test Set Results")
        
        # Save metrics
        metrics_path = os.path.join(self.config.logging.log_dir, "test_metrics.txt")
        with open(metrics_path, 'w') as f:
            for key, value in final_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # Log to wandb
        if self.config.logging.use_wandb:
            for key, value in final_metrics.items():
                wandb.log({f'test/{key}': value})
        
        return final_metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train multimodal deepfake detection model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--config", type=str, default="xception", 
                       choices=["xception", "efficientnet"], help="Model configuration")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for preprocessed data")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == "xception":
        model_config = XCEPTION_CONFIG
    else:
        model_config = EFFICIENTNET_CONFIG
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        model_config, args.data_path, args.cache_dir
    )
    
    # Create model
    print("Creating model...")
    model = MultimodalDeepfakeDetector(model_config)
    
    # Create trainer
    trainer = Trainer(model_config, model, train_loader, val_loader, test_loader)
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_f1 = checkpoint['best_val_f1']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 