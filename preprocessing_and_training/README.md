# **Fashion Tagger: ML Model for Multi-Label Fashion Classification**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your_colab_link_here)
![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-orange)
![Status](https://img.shields.io/badge/status-active-green)

This repository documents the end-to-end process of building a multi-label classification model for fashion products. The project leverages PyTorch and PyTorch Lightning to train a model capable of predicting multiple labels for each fashion image, such as category, color, season, and more.

## **Table of Contents**

- [Introduction](#introduction)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## **Introduction**

Fashion Tagger is designed to classify fashion items using a multi-label approach. The model is trained on the **Fashion Product Images Dataset** from Kaggle, which contains over 44,000 images with multiple labels describing each product. This project demonstrates the integration of AI in fashion, enabling automated and accurate tagging of fashion items.

## **Setup**

### **Environment Setup**
To run the notebook locally, ensure you have Python 3.8+ installed. You can set up the environment and install necessary dependencies with:

```bash
pip install torch torchvision pytorch-lightning timm joblib
```

You can also run the project directly in Google Colab with all dependencies pre-installed.

## **Data Preparation**

### **Loading and Cleaning Data**
The dataset is loaded and cleaned to ensure that all images have corresponding labels and are in the correct format for model training. Rare labels are replaced or removed to maintain the dataset's consistency.

### **One-Hot Encoding and Stratification**
The categorical data is one-hot encoded, and the dataset is split into training and validation sets with stratification to ensure balanced distribution of labels.

### **Image Transformation**
Images undergo transformation and augmentation to improve model generalization, with transformations including resizing, normalization, and augmentation.

## **Model Training**

### **Model Architecture**
The model is based on an EfficientNet-B3 backbone, fine-tuned for our multi-label classification task. The architecture includes custom classifier layers designed to output predictions across multiple labels.

### **Training Process**
The training process is managed with PyTorch Lightning, incorporating:
- **Early Stopping**: To avoid overfitting.
- **Learning Rate Scheduling**: For optimal convergence.
- **Checkpointing**: To save the best model based on validation performance.

```python
trainer = pl.Trainer(max_epochs=80, accelerator='gpu', devices=1)
trainer.fit(model, train_loader, val_loader)
```

### **Monitoring and Logging**
Training progress is logged with TensorBoard, allowing for real-time monitoring of metrics such as loss and accuracy.

## **Evaluation**

### **Performance Metrics**
The model's performance is evaluated using class-wise precision, recall, and F1-score. The results are visualized to provide insights into the model's strengths and areas for improvement.

### **External Data Testing**
The model is tested on external images to validate its performance in real-world scenarios, with predictions being displayed alongside the images.

## **Results**

The final model achieves strong accuracy across all label categories, with F1-scores consistently high for primary attributes like gender, category, and color.

**Sample Predictions:**

![Sample Prediction 1](path_to_image)
![Sample Prediction 2](path_to_image)

## **Future Work and Enhancements**

1. **Advanced Pretrained Models**: Plan to integrate foundation models like CLIP, Vision Transformer (ViT), and DINO for even higher accuracy.
2. **Production Deployment**: Deploy the model via Docker and cloud platforms for broader accessibility.
3. **Image Generation with GANs**: Introduce a GAN-based feature to generate and suggest fashion images based on user input.

## **Contributing**

Contributions are welcome! Please submit pull requests for any improvements or bug fixes.

