"""
NeuroStack Training Engine Module
==================================
Implements the complete training pipeline including:
- Custom training loop with MixUp augmentation
- K-Fold Cross-Validation with Out-of-Fold (OOF) predictions
- Learning rate scheduling and early stopping
- Mixed precision training for Colab T4 GPU efficiency
- Comprehensive metrics tracking

Author: NeuroStack Team
Target Accuracy: 99.95%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import time
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp augmentation to a batch of data.
    
    MixUp creates synthetic training examples by blending pairs of images:
    x_mixed = lambda * x_i + (1 - lambda) * x_j
    y_mixed = lambda * y_i + (1 - lambda) * y_j
    
    This forces the model to learn features rather than memorize images,
    improving generalization (critical for 99.95% accuracy).
    
    Args:
        x: Input images (batch_size, C, H, W)
        y: Labels (batch_size,)
        alpha: Beta distribution parameter (0.2 is standard)
        device: Device to place tensors on
        
    Returns:
        mixed_x: Mixed images
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    
    # Random permutation
    index = torch.randperm(batch_size).to(device)
    
    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Calculate loss for MixUp training.
    
    Loss is a weighted combination of the losses for both labels:
    loss = lambda * loss(pred, y_a) + (1 - lambda) * loss(pred, y_b)
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    
    Monitors validation loss and stops training if it doesn't improve
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save if improved
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            if self.verbose:
                print(f"  ✓ Validation improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⚠ No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  🛑 Early stopping triggered")
                return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save the best model state."""
        self.best_model_state = copy.deepcopy(model.state_dict())
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print("  ✓ Restored best model weights")


class Trainer:
    """
    Main training engine for NeuroStack.
    
    Handles:
    - Training loop with optional MixUp
    - Validation and metrics computation
    - Learning rate scheduling
    - Early stopping
    - Mixed precision training (for memory efficiency on T4)
    - Progress tracking and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        use_amp: bool = True,
        patience_early_stop: int = 10,
        patience_lr_scheduler: int = 3,
        lr_scheduler_factor: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on
            learning_rate: Initial learning rate
            use_mixup: Whether to use MixUp augmentation
            mixup_alpha: MixUp alpha parameter
            use_amp: Whether to use automatic mixed precision
            patience_early_stop: Patience for early stopping
            patience_lr_scheduler: Patience for LR reduction
            lr_scheduler_factor: Factor to reduce LR by
            verbose: Whether to print detailed logs
        """
        self.model = model.to(device)
        self.device = device
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_amp = use_amp
        self.verbose = verbose
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=patience_lr_scheduler
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=patience_early_stop,
            mode='min',
            verbose=verbose
        )
        
        # Mixed precision scaler
        if use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Apply MixUp if enabled
            if self.use_mixup and np.random.rand() > 0.5:  # 50% chance to apply
                batch_images, labels_a, labels_b, lam = mixup_data(
                    batch_images, batch_labels, self.mixup_alpha, self.device
                )
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch_images)
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_images)
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    loss.backward()
                    self.optimizer.step()
                
                # For accuracy, use original labels (approximate)
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += (lam * predicted.eq(labels_a).sum().float() + 
                           (1 - lam) * predicted.eq(labels_b).sum().float()).item()
            
            else:
                # Standard training (no MixUp)
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch_images)
                        loss = self.criterion(outputs, batch_labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_images)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            return_predictions: Whether to return predictions and labels
            
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
            predictions: Predicted labels (if return_predictions=True)
            true_labels: Ground truth labels (if return_predictions=True)
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch_images)
                    loss = self.criterion(outputs, batch_labels)
            else:
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
            
            running_loss += loss.item()
            
            # Get predictions
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = running_loss / len(val_loader)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = 100. * accuracy_score(all_labels, all_predictions)
        
        if return_predictions:
            return avg_loss, accuracy, all_predictions, all_labels, all_probabilities
        else:
            return avg_loss, accuracy, None, None, None
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_path: Path to save best model
            
        Returns:
            Dictionary with training history and final metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING STARTED")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"MixUp: {self.use_mixup}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, _, _, _ = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Manual verbose logging for LR changes
            if new_lr != old_lr and self.verbose:
                print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"\n{'='*80}")
                print(f"Early stopping at epoch {epoch+1}")
                print(f"{'='*80}\n")
                break
        
        # Restore best model
        self.early_stopping.restore_best_model(self.model)
        
        # Save model if path provided
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"✓ Model saved to: {save_path}")
        
        # Final validation
        val_loss, val_acc, predictions, labels, probabilities = self.validate(
            val_loader, return_predictions=True
        )
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        cm = confusion_matrix(labels, predictions)
        
        try:
            auc = roc_auc_score(labels, probabilities[:, 1])
        except:
            auc = 0.0
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best Val Accuracy: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"{'='*80}\n")
        
        return {
            'history': self.history,
            'final_metrics': {
                'accuracy': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm
            },
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }


def train_fold(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    fold_idx: int,
    num_epochs: int = 50,
    device: str = 'cuda',
    use_mixup: bool = False,
    save_dir: str = './models'
) -> Tuple[nn.Module, Dict, np.ndarray]:
    """
    Train a single fold.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        fold_idx: Fold index
        num_epochs: Number of epochs
        device: Device to train on
        use_mixup: Whether to use MixUp
        save_dir: Directory to save model
        
    Returns:
        trained_model: Trained model
        metrics: Training metrics
        oof_predictions: Out-of-fold predictions
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    arch_name = model.__class__.__name__
    save_path = os.path.join(save_dir, f"{arch_name}_fold_{fold_idx}.pth")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        use_mixup=use_mixup,
        use_amp=True  # Always use for Colab T4
    )
    
    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    # Get OOF predictions
    oof_predictions = results['probabilities']
    
    return trainer.model, results, oof_predictions


if __name__ == "__main__":
    print("NeuroStack Training Engine Module")
    print("This module should be imported, not run directly.")
    print("\nExample usage:")
    print("""
    from engine import Trainer, train_fold
    from architectures import ResNet50V2Classifier
    
    # Create model
    model = ResNet50V2Classifier(num_classes=2)
    
    # Initialize trainer
    trainer = Trainer(model, use_mixup=True, device='cuda')
    
    # Train
    results = trainer.train(train_loader, val_loader, num_epochs=50)
    """)
