# Pneumonia Detection with Grad-CAM Visualization

A deep learning-based medical decision support system that classifies chest X-ray images as **Normal** or **Pneumonia** and provides visual explanations using Grad-CAM.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project uses a fine-tuned **ConvNeXt** model to detect pneumonia from chest X-ray images. The model not only provides predictions but also generates **Grad-CAM heatmaps** to visualize which regions of the X-ray influenced the decision.

## Features

- **Binary Classification**: Normal vs Pneumonia detection
- **Pre-trained ConvNeXt**: Transfer learning with ImageNet weights
- **Focal Loss**: Handles class imbalance in the dataset
- **Grad-CAM Visualization**: Explainable AI for medical diagnosis
- **Early Stopping**: Prevents overfitting during training

## Project Structure

```
pneumonia_detector_project/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── config.py        # Hyperparameters and settings
│   ├── dataset.py       # Data loading and augmentation
│   ├── model.py         # ConvNeXt model definition
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Evaluation and metrics
│   └── gradcam.py       # Grad-CAM visualization
├── outputs/
│   └── models/          # Saved model weights
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/keremoztopuz/pneumonia_detector_project.git
cd pneumonia_detector_project

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and extract it to the `data/` folder.

## Usage

### Training

```bash
python -m src.train
```

### Evaluation

```bash
python -m src.evaluate
```

### Grad-CAM Visualization

```bash
python -m src.gradcam
```

## Model Architecture

- **Base Model**: ConvNeXt-Tiny (pretrained on ImageNet)
- **Classifier**: Modified for binary classification
- **Dropout**: 0.2
- **Input Size**: 224x224

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.05 |
| Batch Size | 32 |
| Epochs | 50 |
| Patience | 10 |
| Loss Function | Focal Loss (γ=2) |

## Results

After training, the model outputs:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Confusion Matrix** visualization
- **Grad-CAM heatmaps** for interpretability

## Technologies

- Python
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- seaborn

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)

