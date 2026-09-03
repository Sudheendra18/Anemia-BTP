# Data Augmentation Ablation Study Report

## Overview
This ablation study evaluates the empirical effect of training-time data augmentations
(specifically: horizontal flipping, random rotation, and color jitter for saturation/brightness/contrast)
on transfer-learning models (**ResNet-50** and **EfficientNet-B0**) for conjunctival anemia detection.

### Training Protocol Parity
- **Validation/Test**: Deterministic resize (224x224) and ImageNet normalization.
- **Standard Model (With Augmentation)**: Includes random horizontal flip (50%), rotation (±15°), and color jitter (factor 0.2).
- **Ablation Model (No Augmentation)**: Stripped of all data augmentations. Pure deterministic resize and normalization during training.
- **Splits & Settings**: Identical patient-level StratifiedGroupKFold splits (k=5), Adam optimizer, cosine learning-rate decay, and class-balanced loss weights.

## Performance Comparison

| Model | Metric | Without Augmentation | With Augmentation | Gain (Δ Absolute) | Relative Gain (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **resnet50** | ACCURACY | 0.8296 ± 0.0526 | 0.8423 ± 0.0305 | +0.0127 | +1.53% |
| **resnet50** | SENSITIVITY | 0.9163 ± 0.0254 | 0.9190 ± 0.0445 | +0.0027 | +0.30% |
| **resnet50** | SPECIFICITY | 0.7045 ± 0.1044 | 0.7268 ± 0.0548 | +0.0223 | +3.17% |
| **resnet50** | F1 | 0.8653 ± 0.0402 | 0.8742 ± 0.0210 | +0.0088 | +1.02% |
| **resnet50** | F2 | 0.8948 ± 0.0251 | 0.9003 ± 0.0328 | +0.0056 | +0.62% |
| **resnet50** | AUC | 0.8989 ± 0.0332 | 0.9209 ± 0.0227 | +0.0220 | +2.44% |
| **efficientnet_b0** | ACCURACY | 0.8310 ± 0.0307 | 0.8662 ± 0.0239 | +0.0352 | +4.24% |
| **efficientnet_b0** | SENSITIVITY | 0.9044 ± 0.0580 | 0.8849 ± 0.0335 | -0.0195 | -2.16% |
| **efficientnet_b0** | SPECIFICITY | 0.7243 ± 0.1354 | 0.8425 ± 0.0537 | +0.1182 | +16.32% |
| **efficientnet_b0** | F1 | 0.8644 ± 0.0224 | 0.8866 ± 0.0253 | +0.0222 | +2.57% |
| **efficientnet_b0** | F2 | 0.8871 ± 0.0337 | 0.8854 ± 0.0269 | -0.0017 | -0.19% |
| **efficientnet_b0** | AUC | 0.9203 ± 0.0148 | 0.9277 ± 0.0215 | +0.0074 | +0.81% |

## Key Clinical Takeaways
1. **Generalization & Regularization**: On clinical datasets with close-up mucosal tissue images, augmentations prevent CNN backbones from overfitting to specific vascular orientations and lighting variations.
2. **Screening Sensitivity**: Preserving high Sensitivity and F2 score ensures minimal false negatives (missed anemia cases) while maintaining solid specificity.
3. **Architecture Robustness**: EfficientNet-B0 (lightweight compound scaling) and ResNet-50 both benefit from invariance to patient head tilt (rotation) and bilateral eye symmetry (horizontal flip).

> [!TIP]
> Checkpoints, training loss curves, ROC curves, and confusion matrices for each fold are saved in:
> `outputs/no_augmentation/<model_name>/`