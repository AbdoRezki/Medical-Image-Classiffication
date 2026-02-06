# Chest X-Ray Pneumonia Detection

A deep learning system for automated pneumonia detection from chest X-ray images using transfer learning and interpretable AI techniques.

## 🎯 Project Overview

This project demonstrates:
- **Transfer Learning**: Fine-tuning pre-trained models (ResNet50, EfficientNetB0)
- **Medical Imaging**: Handling real-world healthcare data
- **Model Interpretability**: Grad-CAM visualizations to explain predictions
- **Production Deployment**: Web interface for inference
- **Best Practices**: Proper train/val/test splits, class imbalance handling, comprehensive metrics

## 📊 Dataset

Using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle:
- **Training**: 5,216 images (1,341 normal, 3,875 pneumonia)
- **Validation**: 16 images (8 normal, 8 pneumonia)
- **Test**: 624 images (234 normal, 390 pneumonia)

## 📁 Project Structure

```
medical-image-classifier/
├── data/                          # Dataset directory
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
├── models/                        # Saved models
│   └── best_model.h5
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_experiments.ipynb
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── model.py                  # Model architectures
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── gradcam.py                # Grad-CAM implementation
├── deployment/                    # Deployment files
│   ├── app.py                    # Streamlit web app
│   └── requirements_deploy.txt
├── tests/                        # Unit tests
│   └── test_model.py
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## 🔬 Technical Details

### Models Implemented
- **ResNet50**: Deep residual network, excellent for medical imaging
- **EfficientNetB0**: Efficient compound scaling, better accuracy/parameter ratio

### Techniques Used
- **Data Augmentation**: Rotation, width/height shift, zoom, horizontal flip
- **Transfer Learning**: ImageNet pre-trained weights
- **Class Weights**: Handle imbalanced dataset (3:1 pneumonia to normal ratio)
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: ReduceLROnPlateau for better convergence
- **Grad-CAM**: Visualize which regions influence predictions

### Performance Metrics
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- ROC-AUC Score
- Classification Report


## 🎨 Model Interpretability

The project includes Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize:
- Which regions of the X-ray the model focuses on
- Validation that the model looks at clinically relevant areas (lungs)
- Building trust in model predictions

## 🚀 Deployment

The Streamlit app allows:
- Upload chest X-ray images
- Get predictions with confidence scores
- View Grad-CAM heatmaps showing decision reasoning
- Easy sharing via Streamlit Cloud
