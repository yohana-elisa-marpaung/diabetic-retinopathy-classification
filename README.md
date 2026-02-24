# Diabetic Retinopathy Classification using EfficientNetB3

## Project Overview
An end-to-end Deep Learning system to classify Diabetic Retinopathy (DR) severity from retinal fundus OCT images using EfficientNetB3 architecture. **Final Result: 95.7% accuracy** on 10,000 images across 5 classes --- surpassing previous benchmark by 10.7%.

## Classes
- Class 0: Healthy (No DR)
- Class 1: Mild Diabetic Retinopathy
- Class 2: Moderate Diabetic Retinopathy
- Class 3: Severe Diabetic Retinopathy
- Class 4: Proliferative Diabetic Retinopathy

## Model Performance
| Metric | Score |
|-----------|-------|
| Accuracy | 95.70% |
| Precision | 95.70% |
| Recall | 95.70% |
| F1-Score | 95.70% |

## Tech Stack
- Python 3.x
- PyTorch + TensorFlow/Keras
- EfficientNetB3 (Transfer Learning from ImageNet)
- Albumentations (Data Augmentation)
- Scikit-learn (Evaluation)
- NumPy, Pandas, Matplotlib, Seaborn

## Key Techniques
- Transfer Learning with EfficientNetB3 pretrained on ImageNet
- Data Augmentation: Horizontal Flip, Rotation, Brightness/Contrast, Gaussian Noise
- Class Imbalance Handling: Weighted CrossEntropyLoss + WeightedRandomSampler
- Optimizer: AdamW with Weight Decay
- Scheduler: Cosine Annealing Warm Restarts
- Test Time Augmentation (TTA)
- Gradient Accumulation + Gradient Clipping

## Dataset
- Source: Kaggle - Diabetic Retinopathy Detection by Sandeep Kumar
- Total: 10,000 OCT fundus images (2,000 per class)
- Image size: 256x256 pixels (resized to 224x224)

## Author
Yohana Elisa Marpaung
Physics Graduate - Universitas Sumatera Utara, 2025
Contact: yohanaelisamarpaung@gmail.com
