# Checkpoint-based Training Strategy for Colab Free Tier
# This allows you to resume training across multiple sessions

import os
import json
import pickle
import numpy as np

class TrainingCheckpoint:
    """
    Manages training checkpoints to resume across Colab sessions.
    
    Saves:
    - Trained models
    - OOF predictions
    - Training progress
    - Configuration
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (should be in Google Drive)
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.state_file = os.path.join(checkpoint_dir, "training_state.json")
        self.oof_file = os.path.join(checkpoint_dir, "oof_predictions.pkl")
        
    def save_state(
        self,
        completed_folds_resnet: list,
        completed_folds_densenet: list,
        oof_predictions_resnet: list,
        oof_predictions_densenet: list,
        oof_labels: np.ndarray
    ):
        """
        Save training state.
        
        Args:
            completed_folds_resnet: List of completed ResNet fold indices
            completed_folds_densenet: List of completed DenseNet fold indices
            oof_predictions_resnet: List of OOF predictions from ResNet
            oof_predictions_densenet: List of OOF predictions from DenseNet
            oof_labels: Ground truth labels
        """
        # Save state
        state = {
            'completed_folds_resnet': completed_folds_resnet,
            'completed_folds_densenet': completed_folds_densenet,
            'total_folds': 5
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save OOF predictions
        oof_data = {
            'oof_predictions_resnet': oof_predictions_resnet,
            'oof_predictions_densenet': oof_predictions_densenet,
            'oof_labels': oof_labels
        }
        
        with open(self.oof_file, 'wb') as f:
            pickle.dump(oof_data, f)
        
        print(f"✓ Checkpoint saved")
        print(f"  ResNet folds completed: {completed_folds_resnet}")
        print(f"  DenseNet folds completed: {completed_folds_densenet}")
    
    def load_state(self):
        """
        Load training state from checkpoint.
        
        Returns:
            Dictionary with training state, or None if no checkpoint exists
        """
        if not os.path.exists(self.state_file):
            print("No checkpoint found - starting fresh")
            return None
        
        # Load state
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        
        # Load OOF predictions
        if os.path.exists(self.oof_file):
            with open(self.oof_file, 'rb') as f:
                oof_data = pickle.load(f)
            state.update(oof_data)
        
        print(f"✓ Checkpoint loaded")
        print(f"  ResNet folds completed: {state['completed_folds_resnet']}")
        print(f"  DenseNet folds completed: {state['completed_folds_densenet']}")
        
        return state
    
    def get_next_fold(self, model_type: str, completed_folds: list, total_folds: int = 5):
        """
        Get the next fold to train.
        
        Args:
            model_type: 'resnet' or 'densenet'
            completed_folds: List of completed fold indices
            total_folds: Total number of folds
            
        Returns:
            Next fold index, or None if all folds are complete
        """
        for fold_idx in range(total_folds):
            if fold_idx not in completed_folds:
                return fold_idx
        return None
    
    def is_training_complete(self, state: dict, total_folds: int = 5):
        """
        Check if all training is complete.
        
        Args:
            state: Training state dictionary
            total_folds: Total number of folds
            
        Returns:
            True if all folds are complete
        """
        resnet_complete = len(state['completed_folds_resnet']) == total_folds
        densenet_complete = len(state['completed_folds_densenet']) == total_folds
        
        return resnet_complete and densenet_complete


# Example usage cell for Colab notebook
EXAMPLE_USAGE = """
# ========================================
# RESUMABLE TRAINING CELL
# Run this instead of the original training cells
# ========================================

from checkpoint_training import TrainingCheckpoint

# Initialize checkpoint manager (saves to Google Drive)
checkpoint = TrainingCheckpoint(
    checkpoint_dir="/content/drive/MyDrive/neurostack_outputs/checkpoints"
)

# Load existing progress (if any)
state = checkpoint.load_state()

if state is None:
    # No checkpoint - start fresh
    completed_folds_resnet = []
    completed_folds_densenet = []
    oof_predictions_resnet = []
    oof_predictions_densenet = []
    oof_labels = None
else:
    # Resume from checkpoint
    completed_folds_resnet = state['completed_folds_resnet']
    completed_folds_densenet = state['completed_folds_densenet']
    oof_predictions_resnet = state['oof_predictions_resnet']
    oof_predictions_densenet = state['oof_predictions_densenet']
    oof_labels = state['oof_labels']

print("="*80)
print("TRAINING PROGRESS")
print("="*80)
print(f"ResNet: {len(completed_folds_resnet)}/5 folds complete")
print(f"DenseNet: {len(completed_folds_densenet)}/5 folds complete")
print("="*80)

# Check if training is complete
if checkpoint.is_training_complete({'completed_folds_resnet': completed_folds_resnet, 
                                     'completed_folds_densenet': completed_folds_densenet}):
    print("\\n🎉 All training complete! Proceed to meta-learner.")
else:
    print("\\n▶️ Continuing training...")

# ========================================
# TRAIN RESNET (Resume from checkpoint)
# ========================================

# Get next ResNet fold to train
next_fold = checkpoint.get_next_fold('resnet', completed_folds_resnet)

if next_fold is not None:
    print(f"\\n{'='*80}")
    print(f"TRAINING RESNET - FOLD {next_fold + 1}/5")
    print(f"{'='*80}")
    
    # Create data loaders
    train_loader, val_loader = pipeline.create_fold_datasets(
        fold_idx=next_fold,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )
    
    # Create and train model
    model = ensemble.create_base_model('resnet', pretrained=True)
    
    trained_model, results, oof_preds = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=next_fold,
        num_epochs=CONFIG['num_epochs'],
        device=CONFIG['device'],
        use_mixup=CONFIG['use_mixup'],
        save_dir=MODELS_DIR
    )
    
    # Save progress
    ensemble.add_trained_model(trained_model, 'resnet', next_fold)
    oof_predictions_resnet.append(oof_preds)
    completed_folds_resnet.append(next_fold)
    
    if oof_labels is None:
        oof_labels = results['labels']
    
    # Save checkpoint
    checkpoint.save_state(
        completed_folds_resnet,
        completed_folds_densenet,
        oof_predictions_resnet,
        oof_predictions_densenet,
        oof_labels
    )
    
    print(f"\\n✓ ResNet Fold {next_fold + 1} completed and saved")
else:
    print("\\n✓ All ResNet folds complete")

# ========================================
# TRAIN DENSENET (Resume from checkpoint)
# ========================================

# Get next DenseNet fold to train
next_fold = checkpoint.get_next_fold('densenet', completed_folds_densenet)

if next_fold is not None:
    print(f"\\n{'='*80}")
    print(f"TRAINING DENSENET - FOLD {next_fold + 1}/5")
    print(f"{'='*80}")
    
    # Create data loaders
    train_loader, val_loader = pipeline.create_fold_datasets(
        fold_idx=next_fold,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )
    
    # Create and train model
    model = ensemble.create_base_model('densenet', pretrained=True)
    
    trained_model, results, oof_preds = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=next_fold,
        num_epochs=CONFIG['num_epochs'],
        device=CONFIG['device'],
        use_mixup=CONFIG['use_mixup'],
        save_dir=MODELS_DIR
    )
    
    # Save progress
    ensemble.add_trained_model(trained_model, 'densenet', next_fold)
    oof_predictions_densenet.append(oof_preds)
    completed_folds_densenet.append(next_fold)
    
    # Save checkpoint
    checkpoint.save_state(
        completed_folds_resnet,
        completed_folds_densenet,
        oof_predictions_resnet,
        oof_predictions_densenet,
        oof_labels
    )
    
    print(f"\\n✓ DenseNet Fold {next_fold + 1} completed and saved")
else:
    print("\\n✓ All DenseNet folds complete")

# ========================================
# SUMMARY
# ========================================

print("\\n" + "="*80)
print("SESSION SUMMARY")
print("="*80)
print(f"ResNet: {len(completed_folds_resnet)}/5 folds complete")
print(f"DenseNet: {len(completed_folds_densenet)}/5 folds complete")

if checkpoint.is_training_complete({'completed_folds_resnet': completed_folds_resnet, 
                                     'completed_folds_densenet': completed_folds_densenet}):
    print("\\n🎉 ALL TRAINING COMPLETE!")
    print("   Next: Run the meta-learner cell")
else:
    print("\\n⏸️ Training paused - progress saved to Google Drive")
    print("   To continue: Run this cell again in a new session")
print("="*80)
"""

if __name__ == "__main__":
    print("Checkpoint Training Module")
    print("\nThis module allows you to:")
    print("- Save training progress after each fold")
    print("- Resume training across multiple Colab sessions")
    print("- Avoid losing work when Colab disconnects")
    print("\nSee EXAMPLE_USAGE for implementation")
