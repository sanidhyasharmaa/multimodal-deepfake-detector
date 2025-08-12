"""
Metrics and evaluation utilities for deepfake detection
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from typing import List, Tuple, Dict, Optional
import os


class DeepfakeMetrics:
    """Metrics calculator for deepfake detection"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """
        Update metrics with batch predictions
        
        Args:
            preds: Predicted labels (0 or 1)
            targets: True labels (0 or 1)
            probs: Predicted probabilities (optional)
        """
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary containing all computed metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1': f1_score(targets, predictions, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities are available
        if len(self.probabilities) > 0:
            probabilities = np.array(self.probabilities)
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.targets, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(self.targets, self.predictions, target_names=['Real', 'Fake'])


class MetricsVisualizer:
    """Visualization utilities for metrics"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, train_metrics: Dict[str, List[float]], 
                           val_metrics: Dict[str, List[float]], 
                           save_path: Optional[str] = None):
        """
        Plot training curves
        
        Args:
            train_metrics: Training metrics over epochs
            val_metrics: Validation metrics over epochs
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(train_metrics['loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, train_metrics['accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(epochs, train_metrics['f1'], 'b-', label='Training F1')
        axes[1, 0].plot(epochs, val_metrics['f1'], 'r-', label='Validation F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # ROC-AUC
        if 'roc_auc' in train_metrics and 'roc_auc' in val_metrics:
            axes[1, 1].plot(epochs, train_metrics['roc_auc'], 'b-', label='Training ROC-AUC')
            axes[1, 1].plot(epochs, val_metrics['roc_auc'], 'r-', label='Validation ROC-AUC')
            axes[1, 1].set_title('ROC-AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('ROC-AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_modality_comparison(self, modality_metrics: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None):
        """
        Plot comparison of different modalities
        
        Args:
            modality_metrics: Dictionary of metrics for each modality
            save_path: Path to save the plot (optional)
        """
        modalities = list(modality_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(modalities)
        
        for i, modality in enumerate(modalities):
            values = [modality_metrics[modality].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=modality)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison Across Modalities')
        ax.set_xticks(x + width * (len(modalities) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate all metrics for given predictions
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float], title: str = "Metrics Summary"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        print(f"{metric.upper():<15}: {value:.4f}")
    
    print(f"{'='*50}")


def save_metrics_to_file(metrics: Dict[str, float], filepath: str):
    """
    Save metrics to a text file
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the file
    """
    with open(filepath, 'w') as f:
        f.write("Deepfake Detection Metrics\n")
        f.write("=" * 30 + "\n\n")
        
        for metric, value in metrics.items():
            f.write(f"{metric.upper():<15}: {value:.4f}\n") 