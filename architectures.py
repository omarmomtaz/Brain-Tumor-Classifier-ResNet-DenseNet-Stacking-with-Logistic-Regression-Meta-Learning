"""
NeuroStack Model Architectures Module
======================================
Implements the Stacked Heterogeneous Ensemble for brain tumor classification.

Architecture Overview:
- Level 1 (Base Models): ResNet50V2 + DenseNet121
- Level 2 (Meta-Learner): Logistic Regression

Each base model is a specialist:
- ResNet50V2: Captures global structural features (tumor shape/boundaries)
- DenseNet121: Captures local textural features (tissue patterns)

The meta-learner combines their predictions to make the final decision.

Author: NeuroStack Team
Target Accuracy: 99.95%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


class ResNet50V2Classifier(nn.Module):
    """
    ResNet50V2 base model for brain tumor classification.
    
    Role: Capture global structural features (tumor shape, boundaries, spatial relationships)
    
    ResNet's residual connections allow information to flow through many layers,
    making it excellent at learning hierarchical shape features.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize ResNet50V2 classifier.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use ImageNet pre-trained weights
            dropout_rate: Dropout probability for regularization
            freeze_backbone: If True, freeze backbone weights (only train head)
        """
        super(ResNet50V2Classifier, self).__init__()
        
        # Load pre-trained ResNet50
        # Note: PyTorch doesn't have ResNet50V2, but ResNet50 is very similar
        # V2 uses pre-activation, but for transfer learning, standard ResNet50 works excellently
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified (for fine-tuning strategy)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        # Note: ResNet backbone already includes pooling, so features are already flattened
        self.head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Pass through classification head
        logits = self.head(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (before final classification layer).
        
        Useful for visualization or feature analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        features = self.backbone(x)
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 base model for brain tumor classification.
    
    Role: Capture local textural features (fine-grained tissue patterns)
    
    DenseNet's dense connections preserve fine-grain details throughout
    the network, making it excellent at detecting subtle texture differences.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize DenseNet121 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            dropout_rate: Dropout probability for regularization
            freeze_backbone: If True, freeze backbone weights
        """
        super(DenseNet121Classifier, self).__init__()
        
        # Load pre-trained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier.in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone.features(x)
        features = F.relu(features)
        
        # Pass through classification head
        logits = self.head(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (before final classification layer).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        features = self.backbone.features(x)
        features = F.relu(features)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class StackedEnsemble:
    """
    Stacked Ensemble Meta-Learner.
    
    This is Level 2 of our architecture. It takes the probability predictions
    from multiple base models and learns the optimal way to combine them.
    
    Uses Logistic Regression as the meta-learner (fast, interpretable, effective).
    """
    
    def __init__(
        self,
        n_classes: int = 2,
        use_scaler: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the stacked ensemble.
        
        Args:
            n_classes: Number of classes
            use_scaler: Whether to standardize features before meta-learning
            random_state: Random seed for reproducibility
        """
        self.n_classes = n_classes
        self.use_scaler = use_scaler
        self.random_state = random_state
        
        # Meta-learner
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'  # Handle any class imbalance
        )
        
        # Feature scaler
        if use_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        self.is_fitted = False
    
    def fit(
        self,
        base_predictions: np.ndarray,
        true_labels: np.ndarray
    ):
        """
        Train the meta-learner on Out-of-Fold (OOF) predictions.
        
        CRITICAL: base_predictions must come from OOF predictions to prevent
        data leakage. Never train the meta-learner on predictions from the
        same data used to train the base models.
        
        Args:
            base_predictions: Array of shape (n_samples, n_base_models * n_classes)
                            e.g., for 2 base models and 2 classes: (n_samples, 4)
            true_labels: Ground truth labels of shape (n_samples,)
        """
        # Scale features if using scaler
        if self.scaler is not None:
            base_predictions_scaled = self.scaler.fit_transform(base_predictions)
        else:
            base_predictions_scaled = base_predictions
        
        # Train meta-model
        self.meta_model.fit(base_predictions_scaled, true_labels)
        self.is_fitted = True
        
        print(f"✓ Meta-learner trained on {len(true_labels)} samples")
        print(f"  Feature shape: {base_predictions.shape}")
        print(f"  Meta-model: Logistic Regression")
    
    def predict_proba(
        self,
        base_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities using the meta-learner.
        
        Args:
            base_predictions: Array of shape (n_samples, n_base_models * n_classes)
            
        Returns:
            Predicted probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner is not fitted. Call fit() first.")
        
        # Scale features if using scaler
        if self.scaler is not None:
            base_predictions_scaled = self.scaler.transform(base_predictions)
        else:
            base_predictions_scaled = base_predictions
        
        # Predict probabilities
        probas = self.meta_model.predict_proba(base_predictions_scaled)
        
        return probas
    
    def predict(
        self,
        base_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels using the meta-learner.
        
        Args:
            base_predictions: Array of shape (n_samples, n_base_models * n_classes)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        probas = self.predict_proba(base_predictions)
        return np.argmax(probas, axis=1)
    
    def save(self, filepath: str):
        """Save the meta-learner to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted meta-learner.")
        
        state = {
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'n_classes': self.n_classes,
            'use_scaler': self.use_scaler,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ Meta-learner saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load a saved meta-learner from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.meta_model = state['meta_model']
        self.scaler = state['scaler']
        self.n_classes = state['n_classes']
        self.use_scaler = state['use_scaler']
        self.is_fitted = state['is_fitted']
        
        print(f"✓ Meta-learner loaded from: {filepath}")


class NeuroStackEnsemble:
    """
    Complete NeuroStack ensemble system.
    
    Manages multiple base models (ResNet + DenseNet) across K folds
    and the meta-learner that combines them.
    
    This is the production-ready interface for training and inference.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        num_classes: int = 2,
        device: str = 'cuda'
    ):
        """
        Initialize the NeuroStack ensemble.
        
        Args:
            n_folds: Number of cross-validation folds
            num_classes: Number of output classes
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.n_folds = n_folds
        self.num_classes = num_classes
        self.device = device
        
        # Storage for base models
        self.resnet_models = []  # Will store n_folds ResNet models
        self.densenet_models = []  # Will store n_folds DenseNet models
        
        # Meta-learner
        self.meta_learner = StackedEnsemble(n_classes=num_classes)
        
        print(f"✓ NeuroStack Ensemble initialized")
        print(f"  Folds: {n_folds}")
        print(f"  Device: {device}")
        print(f"  Base models per fold: 2 (ResNet50 + DenseNet121)")
        print(f"  Total base models: {n_folds * 2}")
    
    def create_base_model(
        self,
        model_type: str,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ) -> nn.Module:
        """
        Create a single base model.
        
        Args:
            model_type: 'resnet' or 'densenet'
            pretrained: Whether to use ImageNet weights
            freeze_backbone: Whether to freeze backbone
            
        Returns:
            Initialized model
        """
        if model_type.lower() == 'resnet':
            model = ResNet50V2Classifier(
                num_classes=self.num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
        elif model_type.lower() == 'densenet':
            model = DenseNet121Classifier(
                num_classes=self.num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def add_trained_model(
        self,
        model: nn.Module,
        model_type: str,
        fold_idx: int
    ):
        """
        Add a trained base model to the ensemble.
        
        Args:
            model: Trained PyTorch model
            model_type: 'resnet' or 'densenet'
            fold_idx: Which fold this model was trained on
        """
        if model_type.lower() == 'resnet':
            self.resnet_models.append({
                'model': model,
                'fold_idx': fold_idx
            })
        elif model_type.lower() == 'densenet':
            self.densenet_models.append({
                'model': model,
                'fold_idx': fold_idx
            })
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_base_models(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_labels: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions from all base models.
        
        Args:
            dataloader: DataLoader with images
            return_labels: Whether to return ground truth labels
            
        Returns:
            base_predictions: Array of shape (n_samples, n_base_models * n_classes)
            labels: Ground truth labels if return_labels=True
        """
        all_predictions = []
        all_labels = []
        
        # Combine all base models
        all_models = self.resnet_models + self.densenet_models
        
        if len(all_models) == 0:
            raise ValueError("No base models added to the ensemble.")
        
        # Get predictions from each base model
        for model_info in all_models:
            model = model_info['model']
            model.eval()
            
            model_preds = []
            
            with torch.no_grad():
                for batch_images, batch_labels in dataloader:
                    batch_images = batch_images.to(self.device)
                    
                    # Get logits
                    logits = model(batch_images)
                    
                    # Convert to probabilities
                    probas = F.softmax(logits, dim=1)
                    
                    model_preds.append(probas.cpu().numpy())
                    
                    # Collect labels (only once)
                    if return_labels and len(all_labels) == 0:
                        all_labels.append(batch_labels.numpy())
            
            # Concatenate all batches for this model
            model_preds = np.vstack(model_preds)
            all_predictions.append(model_preds)
        
        # Combine predictions from all models
        # Shape: (n_samples, n_models * n_classes)
        base_predictions = np.hstack(all_predictions)
        
        if return_labels:
            labels = np.concatenate(all_labels)
            return base_predictions, labels
        else:
            return base_predictions, None
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make final predictions using the complete ensemble.
        
        Args:
            dataloader: DataLoader with images
            
        Returns:
            predictions: Predicted class labels
            probabilities: Predicted class probabilities
        """
        # Get base model predictions
        base_predictions, _ = self.predict_base_models(dataloader)
        
        # Use meta-learner for final prediction
        probabilities = self.meta_learner.predict_proba(base_predictions)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def save_ensemble(self, save_dir: str):
        """
        Save the complete ensemble to disk.
        
        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save ResNet models
        for idx, model_info in enumerate(self.resnet_models):
            path = os.path.join(save_dir, f"resnet_fold_{model_info['fold_idx']}.pth")
            torch.save(model_info['model'].state_dict(), path)
        
        # Save DenseNet models
        for idx, model_info in enumerate(self.densenet_models):
            path = os.path.join(save_dir, f"densenet_fold_{model_info['fold_idx']}.pth")
            torch.save(model_info['model'].state_dict(), path)
        
        # Save meta-learner
        meta_path = os.path.join(save_dir, "meta_learner.pkl")
        self.meta_learner.save(meta_path)
        
        print(f"✓ Complete ensemble saved to: {save_dir}")


def get_model_summary(model: nn.Module) -> dict:
    """
    Get a summary of model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }


if __name__ == "__main__":
    # Example usage
    print("NeuroStack Architectures Module")
    print("This module should be imported, not run directly.")
    print("\nExample usage:")
    print("""
    from architectures import NeuroStackEnsemble
    
    # Initialize ensemble
    ensemble = NeuroStackEnsemble(n_folds=5, device='cuda')
    
    # Create a base model
    resnet = ensemble.create_base_model('resnet', pretrained=True)
    
    # After training, add it to the ensemble
    ensemble.add_trained_model(resnet, 'resnet', fold_idx=0)
    
    # Make predictions
    predictions, probabilities = ensemble.predict(test_loader)
    """)
