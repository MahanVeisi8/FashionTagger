# **Fashion Tagger: ML Model for Multi-Label Fashion Classification**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OcDUrJpxSpI-p2jbxc9P0KaU_3wgtvIK?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-orange)
![Status](https://img.shields.io/badge/status-active-green)

## Setup

**Running the Notebook in Google Colab**
- The notebook is designed for easy execution in Google Colab, requiring no additional setup other than a Google account and internet access.😊

The code is designed to run in a Python environment with essential machine learning and simulation libraries. You can execute the notebook directly in Google Colab using the badge link provided, which includes a pre-configured environment with all necessary dependencies.

## **Data Preparation**

### **Data Collection**

To ensure the highest quality and accuracy in fashion image classification, we utilized the full **Fashion Product Images Dataset** from Kaggle, which includes over 44,000 images. Unlike the smaller, compressed version, this full dataset (25 GB) offers high-resolution images that are essential for accurately detecting attributes such as colors and fine details in fashion items.

Given the dataset's large size, processing was performed on Google Colab Pro with T4 GPU support, ensuring sufficient disk space and processing power for efficient handling and accelerated model training.





### **Dataset Overview and Initial Setup**

After obtaining the full Fashion Product Images Dataset, we loaded the dataset into a Pandas DataFrame, ensuring that all image paths and associated metadata were correctly mapped. Below is a preview of the dataset:

![Dataset Overview](image1.png)

Given the vast amount of data, it was crucial to verify and clean the dataset to ensure the integrity of our analysis.

### **Dataset Cleaning**

To ensure that our dataset was both complete and accurate, we implemented a cleaning process to remove any entries with missing images. This step was necessary to avoid potential issues during the training phase. After cleaning, the dataset shape was adjusted accordingly, ensuring only valid data was retained.

### **Dataset Analysis**

We performed a thorough analysis of the dataset to understand its structure. This included checking for null values, counting unique values in each categorical column, and reviewing the overall distribution of labels. Below, you can see the unique values across different categories such as gender, masterCategory, subCategory, and others.

We found that the dataset had an imbalance in certain classes, with some categories having significantly fewer samples. This imbalance could potentially affect the model's performance.

### **Handling Imbalanced Data**

Given the imbalance, we decided to handle classes with fewer than 500 samples by relabeling them as 'None', effectively treating them as a generic "other" category. This approach allowed us to retain as much data as possible while avoiding the challenges of training on extremely small class sizes. This process reduced the risk of overfitting to rare classes and ensured a more robust model.

![Class Imbalance Visualization](image2.png)

### **Image ID Preparation and Visualization**

To match image IDs with their filenames, we appended the `.jpg` extension to each image ID. We then visualized a sample of images along with their labels to confirm the integrity and quality of our dataset, ensuring that the labels were correctly aligned with the images.






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

