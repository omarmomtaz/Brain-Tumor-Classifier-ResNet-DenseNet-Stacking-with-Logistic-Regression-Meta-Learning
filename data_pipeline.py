"""
NeuroStack Data Pipeline Module
================================
Implements the "99% Protocol" preprocessing pipeline for brain tumor MRI classification.

Pipeline Sequence (Applied to ALL data):
1. Smart Crop (contour-based skull extraction)
2. Resize to 224x224
3. Gaussian Blur (denoise BEFORE contrast enhancement)
4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Channel Replication (1-channel grayscale → 3-channel for transfer learning)
6. Normalization (ImageNet statistics)

Author: NeuroStack Team
Target Accuracy: 99.95%
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import matplotlib.pyplot as plt


class BrainMRIPreprocessor:
    """
    Handles the preprocessing pipeline for brain MRI images.
    
    This class implements the scientifically optimal sequence for maximum
    Signal-to-Noise Ratio (SNR) in tumor detection.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        gaussian_kernel_size: Tuple[int, int] = (3, 3),
        apply_clahe: bool = True,
        apply_gaussian: bool = True
    ):
        """
        Initialize the preprocessor with pipeline parameters.
        
        Args:
            target_size: Final image dimensions (width, height)
            clahe_clip_limit: CLAHE contrast limiting parameter
            clahe_tile_grid_size: CLAHE grid size for local enhancement
            gaussian_kernel_size: Kernel size for Gaussian blur denoising
            apply_clahe: Whether to apply CLAHE enhancement
            apply_gaussian: Whether to apply Gaussian blur
        """
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.apply_clahe = apply_clahe
        self.apply_gaussian = apply_gaussian
        
        # Initialize CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
        
    def smart_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1: Smart Crop - Remove black background using contour detection.
        
        MRI scans typically have 30-40% empty black space around the skull.
        This step eliminates that waste, making the tumor larger and more
        prominent relative to the image size.
        
        Args:
            image: Input grayscale image (H, W) or (H, W, C)
            
        Returns:
            Cropped image containing only the brain region
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold to separate brain from background
        # Use Otsu's method for automatic threshold detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (the skull boundary should be the largest)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No contours found - return original image
            return image
        
        # Find the largest contour (should be the brain)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding (5%) to avoid cutting off edges
        padding = int(0.05 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop to bounding box
        if len(image.shape) == 3:
            cropped = image[y:y+h, x:x+w, :]
        else:
            cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def preprocess_single_image(self, image_path: str) -> np.ndarray:
        """
        Apply the complete preprocessing pipeline to a single image.
        
        Pipeline:
        1. Load as grayscale
        2. Smart Crop (remove background)
        3. Resize to target size
        4. Gaussian Blur (denoise BEFORE CLAHE)
        5. CLAHE (enhance contrast)
        6. Channel Replication (1→3 channels)
        7. Ready for normalization (handled by Dataset class)
        
        Args:
            image_path: Path to the MRI image file
            
        Returns:
            Preprocessed image as numpy array (224, 224, 3)
        """
        # Load image as grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Step 1: Smart Crop
        cropped = self.smart_crop(image)
        
        # Step 2: Resize to target size
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Step 3: Gaussian Blur (denoise BEFORE contrast enhancement)
        # Critical: Denoising before CLAHE prevents magnifying sensor noise
        if self.apply_gaussian:
            blurred = cv2.GaussianBlur(resized, self.gaussian_kernel_size, 0)
        else:
            blurred = resized
        
        # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Enhances local contrast to reveal tumor textures
        if self.apply_clahe:
            enhanced = self.clahe.apply(blurred)
        else:
            enhanced = blurred
        
        # Step 5: Channel Replication (1-channel → 3-channel)
        # This preserves pre-trained ImageNet weights in the first conv layer
        image_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return image_3ch
    
    def visualize_pipeline(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize each step of the preprocessing pipeline for quality assurance.
        
        Args:
            image_path: Path to an example MRI image
            save_path: Optional path to save the visualization
        """
        # Load original
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply each step sequentially
        cropped = self.smart_crop(original)
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, self.gaussian_kernel_size, 0) if self.apply_gaussian else resized
        enhanced = self.clahe.apply(blurred) if self.apply_clahe else blurred
        final_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('NeuroStack Preprocessing Pipeline', fontsize=16, fontweight='bold')
        
        steps = [
            (original, 'Original MRI', 'gray'),
            (cropped, 'After Smart Crop\n(Background Removed)', 'gray'),
            (resized, f'After Resize\n({self.target_size[0]}x{self.target_size[1]})', 'gray'),
            (blurred, 'After Gaussian Blur\n(Denoised)', 'gray'),
            (enhanced, 'After CLAHE\n(Contrast Enhanced)', 'gray'),
            (final_3ch, 'Final (3-Channel)\nReady for Model', None)
        ]
        
        for idx, (img, title, cmap) in enumerate(steps):
            ax = axes[idx // 3, idx % 3]
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Add image statistics
            ax.text(0.02, 0.98, f'Shape: {img.shape}\nMin: {img.min()}\nMax: {img.max()}\nMean: {img.mean():.1f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pipeline visualization saved to: {save_path}")
        
        plt.show()


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI images.
    
    Handles:
    - Binary classification (Meningioma vs Glioma)
    - Preprocessing pipeline integration
    - Optional data augmentation (training only)
    - ImageNet normalization
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        preprocessor: BrainMRIPreprocessor,
        augmentation: Optional[transforms.Compose] = None,
        is_training: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to MRI images
            labels: List of corresponding labels (0=Meningioma, 1=Glioma)
            preprocessor: BrainMRIPreprocessor instance
            augmentation: Optional augmentation transforms (training only)
            is_training: Whether this is a training dataset
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.is_training = is_training
        
        # ImageNet normalization statistics
        # These are the standard values used for transfer learning
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Base transform (always applied)
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a preprocessed image and its label.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (preprocessed_image_tensor, label)
        """
        # Apply preprocessing pipeline
        image = self.preprocessor.preprocess_single_image(self.image_paths[idx])
        label = self.labels[idx]
        
        # Convert to PIL Image for torchvision transforms
        image_pil = Image.fromarray(image)
        
        # Apply augmentation if training
        if self.is_training and self.augmentation is not None:
            image_pil = self.augmentation(image_pil)
        
        # Convert to tensor and normalize
        image_tensor = self.to_tensor(image_pil)
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor, label


class DataPipelineManager:
    """
    Manages the complete data pipeline including:
    - Dataset loading and filtering
    - Train/test split
    - K-Fold cross-validation setup
    - DataLoader creation
    """
    
    def __init__(
        self,
        dataset_root: str,
        target_classes: List[str] = ['meningioma', 'glioma'],
        class_to_idx: Dict[str, int] = {'meningioma': 0, 'glioma': 1},
        n_folds: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize the data pipeline manager.
        
        Args:
            dataset_root: Root directory of the dataset
            target_classes: Classes to include (default: meningioma, glioma)
            class_to_idx: Mapping from class name to numeric label
            n_folds: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
        """
        self.dataset_root = Path(dataset_root)
        self.target_classes = target_classes
        self.class_to_idx = class_to_idx
        self.n_folds = n_folds
        self.random_seed = random_seed
        
        # Initialize preprocessor
        self.preprocessor = BrainMRIPreprocessor()
        
        # Storage for loaded data
        self.train_paths = []
        self.train_labels = []
        self.test_paths = []
        self.test_labels = []
        
    def load_dataset(self) -> Dict[str, any]:
        """
        Load the dataset and filter for target classes only.
        
        Returns:
            Dictionary with dataset statistics
        """
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        # Load training data
        train_dir = self.dataset_root / "Training"
        for class_name in self.target_classes:
            class_dir = train_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Training directory not found: {class_dir}")
            
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            label = self.class_to_idx[class_name]
            
            self.train_paths.extend([str(f) for f in image_files])
            self.train_labels.extend([label] * len(image_files))
            
            print(f"✓ Loaded {len(image_files)} {class_name} images (Training)")
        
        # Load test data
        test_dir = self.dataset_root / "Testing"
        for class_name in self.target_classes:
            class_dir = test_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Test directory not found: {class_dir}")
            
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            label = self.class_to_idx[class_name]
            
            self.test_paths.extend([str(f) for f in image_files])
            self.test_labels.extend([label] * len(image_files))
            
            print(f"✓ Loaded {len(image_files)} {class_name} images (Testing)")
        
        print("=" * 80)
        print(f"TOTAL: {len(self.train_paths)} training, {len(self.test_paths)} test images")
        print("=" * 80)
        
        # Calculate class distribution
        train_class_counts = {cls: self.train_labels.count(idx) for cls, idx in self.class_to_idx.items()}
        test_class_counts = {cls: self.test_labels.count(idx) for cls, idx in self.class_to_idx.items()}
        
        return {
            'train_total': len(self.train_paths),
            'test_total': len(self.test_paths),
            'train_distribution': train_class_counts,
            'test_distribution': test_class_counts
        }
    
    def get_augmentation_transforms(self) -> transforms.Compose:
        """
        Get data augmentation transforms for training.
        
        CRITICAL: Augmentation is ONLY applied to training data.
        Test/validation data must remain "clean" for accurate evaluation.
        
        Allowed augmentations for MRI:
        - Horizontal/Vertical Flip (brain is symmetric)
        - Rotation (±15 degrees)
        - Width/Height Shift
        
        Prohibited:
        - Color Jitter (destroys density information)
        - Extreme rotations (>30 degrees)
        - Excessive shearing
        
        Returns:
            Composed augmentation transforms
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 10% shift
                scale=(0.9, 1.1),      # ±10% zoom
                shear=5                 # Minimal shear
            ),
        ])
    
    def create_fold_datasets(
        self,
        fold_idx: int,
        batch_size: int = 32,
        num_workers: int = 2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for a specific fold in cross-validation.
        
        Args:
            fold_idx: Index of the fold (0 to n_folds-1)
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create stratified K-Fold split
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        
        # Get train/val indices for this fold
        train_paths_array = np.array(self.train_paths)
        train_labels_array = np.array(self.train_labels)
        
        for idx, (train_idx, val_idx) in enumerate(skf.split(train_paths_array, train_labels_array)):
            if idx == fold_idx:
                fold_train_paths = train_paths_array[train_idx].tolist()
                fold_train_labels = train_labels_array[train_idx].tolist()
                fold_val_paths = train_paths_array[val_idx].tolist()
                fold_val_labels = train_labels_array[val_idx].tolist()
                break
        
        # Create datasets
        augmentation = self.get_augmentation_transforms()
        
        train_dataset = BrainTumorDataset(
            fold_train_paths,
            fold_train_labels,
            self.preprocessor,
            augmentation=augmentation,
            is_training=True
        )
        
        val_dataset = BrainTumorDataset(
            fold_val_paths,
            fold_val_labels,
            self.preprocessor,
            augmentation=None,  # NO augmentation on validation
            is_training=False
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nFold {fold_idx + 1}/{self.n_folds}:")
        print(f"  Training samples: {len(fold_train_paths)}")
        print(f"  Validation samples: {len(fold_val_paths)}")
        
        return train_loader, val_loader
    
    def create_test_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 2
    ) -> DataLoader:
        """
        Create DataLoader for the held-out test set.
        
        Args:
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            
        Returns:
            Test DataLoader
        """
        test_dataset = BrainTumorDataset(
            self.test_paths,
            self.test_labels,
            self.preprocessor,
            augmentation=None,  # NO augmentation on test set
            is_training=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nTest set: {len(self.test_paths)} samples")
        
        return test_loader


def verify_preprocessing(dataset_root: str, output_dir: str = "./preprocessing_verification"):
    """
    Utility function to verify preprocessing quality on sample images.
    
    Args:
        dataset_root: Root directory of the dataset
        output_dir: Directory to save verification images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    preprocessor = BrainMRIPreprocessor()
    
    # Get sample images from each class
    for class_name in ['meningioma', 'glioma']:
        class_dir = Path(dataset_root) / "Training" / class_name
        sample_images = list(class_dir.glob("*.jpg"))[:3]  # Get 3 samples
        
        for idx, img_path in enumerate(sample_images):
            output_path = os.path.join(output_dir, f"{class_name}_sample_{idx+1}.png")
            preprocessor.visualize_pipeline(str(img_path), save_path=output_path)


if __name__ == "__main__":
    # Example usage and testing
    print("NeuroStack Data Pipeline Module")
    print("This module should be imported, not run directly.")
    print("\nExample usage:")
    print("""
    from data_pipeline import DataPipelineManager, verify_preprocessing
    
    # Initialize pipeline
    pipeline = DataPipelineManager(
        dataset_root="/content/drive/MyDrive/Dataset/Brain Tumor MRI Dataset - organization. Masoud Nickparvar"
    )
    
    # Load dataset
    stats = pipeline.load_dataset()
    
    # Verify preprocessing quality
    verify_preprocessing(pipeline.dataset_root)
    
    # Create fold loaders
    train_loader, val_loader = pipeline.create_fold_datasets(fold_idx=0, batch_size=32)
    """)
