<div align="center">

```
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗████████╗ █████╗  ██████╗██╗  ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗   ██║   ███████║██║     █████╔╝ 
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║   ██║   ██║  ██║╚██████╗██║  ██╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

**Brain Tumor Classification Engine**  
*ResNet50V2 + DenseNet121 · 5-Fold Stacked Ensemble · 99.88% Accuracy*

---

![Accuracy](https://img.shields.io/badge/Accuracy-99.88%25-brightgreen?style=for-the-badge&labelColor=0a0a0a)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9975-blue?style=for-the-badge&labelColor=0a0a0a)
![Sensitivity](https://img.shields.io/badge/Sensitivity-99.75%25-brightgreen?style=for-the-badge&labelColor=0a0a0a)
![Specificity](https://img.shields.io/badge/Specificity-100.00%25-brightgreen?style=for-the-badge&labelColor=0a0a0a)
![Python](https://img.shields.io/badge/Python-3.10+-orange?style=for-the-badge&logo=python&logoColor=white&labelColor=0a0a0a)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=0a0a0a)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Clinical Context](#clinical-context)
- [Architecture](#architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Author](#author)

---

## Overview

NeuroStack is a production-grade stacked ensemble classifier for brain tumor MRI diagnosis. Given a T1/T2 contrast MRI scan, the system distinguishes **Glioma** (aggressive, malignant) from **Meningioma** (typically benign) with near-clinical-grade precision.

**The system architecture:**
- **Level 1:** 10 independent CNN base models — ResNet50V2 and DenseNet121, each trained across 5 stratified cross-validation folds
- **Level 2:** Logistic Regression meta-learner trained on out-of-fold predictions, producing the final calibrated probability

The ensemble achieves **99.88% accuracy** on a held-out test set of 800 scans — with zero false positives and only one false negative.

---

## Clinical Context

| Tumor Type | Nature | Clinical Pathway |
|---|---|---|
| **Glioma** | Aggressive, infiltrative, malignant | Urgent chemotherapy + radiation |
| **Meningioma** | Typically benign, slow-growing | Surgical + watchful waiting |

Misclassification is not a metric failure — it is a patient harm event. NeuroStack was designed with this constraint as a first-order requirement: **specificity and sensitivity cannot be sacrificed against each other**. The zero-FP result means no meningioma patient was incorrectly routed toward aggressive glioma treatment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│              MRI Scans · 224×224 · RGB normalized               │
│          Augmentation: MixUp + Flip + Rotate + Contrast         │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌─────────────────────┐       ┌─────────────────────┐
│   ResNet50V2 Wing   │       │  DenseNet121 Wing   │
│                     │       │                     │
│  Fold 0 → P(class)  │       │  Fold 0 → P(class)  │
│  Fold 1 → P(class)  │       │  Fold 1 → P(class)  │
│  Fold 2 → P(class)  │       │  Fold 2 → P(class)  │
│  Fold 3 → P(class)  │       │  Fold 3 → P(class)  │
│  Fold 4 → P(class)  │       │  Fold 4 → P(class)  │
│                     │       │                     │
│   5 × [P0, P1]      │       │   5 × [P0, P1]      │
└──────────┬──────────┘       └──────────┬──────────┘
           │                             │
           └──────────┬──────────────────┘
                      │ hstack → shape (N, 20)
                      ▼
          ┌───────────────────────┐
          │    LEVEL 2            │
          │  StandardScaler       │
          │  LogisticRegression   │
          │  (trained on OOF)     │
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   FINAL PREDICTION    │
          │  P(Glioma) · P(Meni)  │
          │  Binary class label   │
          └───────────────────────┘
```

### Why Stacking?

Simple majority voting discards rich probabilistic information. The meta-learner learns *how much to trust each base model* and in what context — a fundamentally more powerful aggregation strategy. Individual fold AUCs range from 0.940–0.961; the ensemble achieves **0.9975**.

### Why Heterogeneous Architectures?

| Architecture | Key Property | Inductive Bias |
|---|---|---|
| **ResNet50V2** | Residual skip connections | Gradient flow · Global texture |
| **DenseNet121** | Dense feature reuse | Fine-grained local features |

Networks that fail differently are more valuable than networks that agree. Their prediction correlation of 0.93–0.99 confirms agreement on easy cases while their failure modes diverge on hard ones — exactly where ensemble gains are realized.

---

## Results

### Core Metrics

| Metric | Value | 95% CI |
|---|---|---|
| **Accuracy** | **99.88%** | [99.30% – 99.98%] |
| **Sensitivity (Recall)** | **99.75%** | [98.60% – 99.96%] |
| **Specificity (TNR)** | **100.00%** | [99.05% – 100.00%] |
| **Precision (PPV)** | **100.00%** | — |
| **Negative Predictive Value** | **99.75%** | — |
| **AUC-ROC** | **0.9975** | — |
| **AUC-PR (Avg Precision)** | **0.9988** | — |
| **F1-Score (Glioma)** | **0.9987** | — |
| **F1-Score (Macro)** | **0.9987** | — |
| **Matthews Corr. Coef.** | **0.9975** | — |
| **Cohen's Kappa** | **0.9975** | — |
| **Brier Score** | **0.0182** | Lower is better |
| **Log Loss** | **0.0598** | Lower is better |
| **Positive Likelihood Ratio** | **∞** | FP = 0 |
| **Negative Likelihood Ratio** | **0.0025** | <0.1 = strong evidence |
| **Diagnostic Odds Ratio** | **∞** | — |
| **Youden's J Index** | **0.9975** | — |
| **Optimal Threshold** | **0.5104** | Youden-J on ROC |

### Confusion Matrix

```
                  Predicted
                  Meningioma    Glioma
Actual  Meningioma    400  │      0       ← 100.00% specificity
        ─────────────────────────────
        Glioma          1  │    399       ← 99.75% sensitivity

        TP=399  TN=400  FP=0  FN=1
```

### Per-Class Report

```
              precision    recall    f1-score   support
Meningioma     0.9975      1.0000     0.9988       400
Glioma         1.0000      0.9975     0.9987       400
─────────────────────────────────────────────────────
macro avg      0.9988      0.9988     0.9987       800
weighted avg   0.9988      0.9988     0.9987       800
```

### Base Model Performance

| Model | AUC-ROC | Accuracy |
|---|---|---|
| ResNet fold 0 | 0.952 | 91.4% |
| ResNet fold 1 | 0.940 | 90.2% |
| ResNet fold 2 | 0.947 | 92.1% |
| ResNet fold 3 | 0.951 | 91.4% |
| ResNet fold 4 | 0.956 | 91.9% |
| DenseNet fold 0 | 0.956 | 93.2% |
| DenseNet fold 1 | 0.951 | 92.2% |
| DenseNet fold 2 | 0.953 | 92.9% |
| DenseNet fold 3 | 0.945 | 91.9% |
| DenseNet fold 4 | 0.961 | 92.9% |
| **NeuroStack Ensemble** | **0.9975** | **99.88%** |

---

## Visualizations

The evaluation engine generates 14 diagnostic plots automatically saved to `neurostack_outputs/evaluation_plots/`.

| Figure | Description |
|---|---|
| `fig1_metrics_dashboard.png` | 8-panel green/yellow metric scorecards |
| `fig2_confusion_matrix.png` | Absolute counts + row-normalized heatmap |
| `fig3_roc_curve.png` | Ensemble ROC vs all 10 base model ROCs |
| `fig4_pr_curve.png` | Precision-Recall with average precision |
| `fig5_probability_distribution.png` | Histogram + violin by true class |
| `fig6_calibration_curve.png` | Reliability diagram + probability histogram |
| `fig7_base_model_comparison.png` | AUC and accuracy bars per fold |
| `fig8_error_analysis.png` | FP/FN profiles + outcome pie + error confidence |
| `fig9_threshold_analysis.png` | Clinical metrics and F1/Youden vs threshold sweep |
| `fig10_model_diversity.png` | Correlation heatmap + ResNet vs DenseNet scatter |
| `fig11_per_class_metrics.png` | Precision/Recall/F1 bar chart per class |
| `fig12_radar_chart.png` | Clinical performance spider chart |
| `fig13_det_curve.png` | Detection Error Tradeoff on normal deviate scale |
| `fig14_metrics_table.png` | Complete text-based metrics reference |

---

## Project Structure

```
neurostack/
│
├── neurostack.ipynb              # Main training + evaluation notebook
│
├── src/
│   ├── data_pipeline.py          # DataPipelineManager, BrainTumorDataset,
│   │                             #   BrainMRIPreprocessor
│   ├── architectures.py          # ResNet50V2Classifier, DenseNet121Classifier,
│   │                             #   StackedEnsemble, NeuroStackEnsemble
│   └── engine.py                 # Trainer, train_fold, EarlyStopping
│
├── outputs/
│   ├── models/
│   │   ├── ResNet50V2Classifier_fold_0.pth  ─┐
│   │   ├── ResNet50V2Classifier_fold_1.pth   │  5 ResNet checkpoints
│   │   ├── ...                               │
│   │   ├── DenseNet121Classifier_fold_0.pth ─┤
│   │   ├── DenseNet121Classifier_fold_1.pth  │  5 DenseNet checkpoints
│   │   ├── ...                               │
│   │   └── meta_learner.pkl      ─────────────┘  Stacking meta-learner
│   │
│   └── evaluation_plots/
│       ├── fig1_metrics_dashboard.png
│       ├── fig2_confusion_matrix.png
│       └── ...                   (14 plots total)
│
└── README.md
```

---

## Quickstart

### Prerequisites

```bash
pip install torch torchvision scikit-learn matplotlib seaborn scipy numpy
```

### Run Inference on a Single Scan

```python
from src.architectures import NeuroStackEnsemble
from src.data_pipeline import BrainMRIPreprocessor
import torch

# Load ensemble
ensemble = NeuroStackEnsemble(n_folds=5, num_classes=2, device='cuda')

# Load all 10 base models
for fold_idx in range(5):
    for arch in ['resnet', 'densenet']:
        name = f"{'ResNet50V2' if arch == 'resnet' else 'DenseNet121'}Classifier_fold_{fold_idx}.pth"
        model = ensemble.create_base_model(arch, pretrained=False)
        model.load_state_dict(torch.load(f"outputs/models/{name}", map_location='cuda'))
        model.eval()
        ensemble.add_trained_model(model, arch, fold_idx)

ensemble.meta_learner.load("outputs/models/meta_learner.pkl")

# Predict
preprocessor = BrainMRIPreprocessor()
image_tensor = preprocessor.process_single("path/to/scan.jpg").unsqueeze(0)

with torch.no_grad():
    prob = ensemble.predict_single(image_tensor)

print(f"P(Glioma): {prob[1]:.4f} | P(Meningioma): {prob[0]:.4f}")
print(f"Prediction: {'Glioma' if prob[1] >= 0.5 else 'Meningioma'}")
```

---

## Training Pipeline

The full training pipeline runs inside `neurostack.ipynb`. Key configuration:

```python
CONFIG = {
    'n_folds':    5,
    'batch_size': 32,
    'num_epochs': 50,
    'use_mixup':  True,
    'device':     'cuda',
    'num_workers': 2,
}

DATASET_ROOT = "/path/to/Brain Tumor MRI Dataset - Masoud Nickparvar"
OUTPUT_DIR   = "/path/to/neurostack_outputs"
```

**Training order:**
1. `DataPipelineManager` creates stratified 5-fold splits
2. `train_fold()` trains each ResNet and DenseNet fold with EarlyStopping
3. Out-of-fold predictions are collected and used to train the meta-learner
4. All 10 `.pth` checkpoints and `meta_learner.pkl` are saved to `OUTPUT_DIR/models/`

---

## Evaluation

Run the evaluation cell after training (or load saved models directly):

```python
# The evaluation cell auto-detects saved models and runs:
# 1. Model reload from checkpoints
# 2. Full test set inference
# 3. 20+ metric computation with 95% CI
# 4. 14 diagnostic visualizations saved to Drive
```

---

## Dataset

**Brain Tumor MRI Dataset** — Masoud Nickparvar  
Available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Meningioma | Glioma | Total |
|---|---|---|---|
| Training | 1,400 | 1,400 | 2,800 |
| Test | 400 | 400 | 800 |
| **Total** | **1,800** | **1,800** | **3,600** |

Classes are perfectly balanced. No data leakage was possible across folds due to patient-level stratification enforced by `DataPipelineManager`.

---

## Author

**Omar Momtaz** — AI Engineer · GenAI Builder · Medical ML

> *"The goal was not to build a model that performs well. The goal was to build a model that a clinician could trust."*

Building at the intersection of **generative AI, computer vision, and embodied intelligence**. NeuroStack is one project in a broader mission to bring rigorous, production-grade machine learning to high-stakes domains.

<div align="center">
<a href="[https://www.upwork.com](https://www.upwork.com/freelancers/~01044b5d38a3457dc1?mp_source=share)">
  <img src="https://img.shields.io/badge/Hire%20on-Upwork-D4AF37?style=for-the-badge&labelColor=8B0000"/>
</a>
<a href="[https://www.linkedin.com](https://www.linkedin.com/in/omar-momtaz-/)">
  <img src="https://img.shields.io/badge/Connect-LinkedIn-00E5FF?style=for-the-badge&labelColor=111111"/>
</a>
<a href="[https://twitter.com](https://x.com/omarmomtaz_main)">
  <img src="https://img.shields.io/badge/Follow-X-D4AF37?style=for-the-badge&labelColor=111111"/>
</a>
</div>

---

<div align="center">

*NeuroStack · Brain Tumor Classification Engine · 2026*  
*Built with PyTorch · scikit-learn · Matplotlib*

</div>
