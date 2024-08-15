# **Fashion Tagger: AI-Powered Fashion Image Labeling**

## Dataset Explanation

Fashion Tagger uses the **Fashion Product Images Dataset** from Kaggle, which contains over 44,000 fashion product images with detailed labels such as category, color, and season. This dataset is crucial for training our AI model to accurately predict fashion item attributes.

- **Dataset Source**: [Fashion Product Images Dataset on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

## Project Overview

Fashion Tagger is a web application that classifies fashion items using an AI model trained on the Kaggle dataset. The model, built using an ensemble of CNNs, predicts labels like category and color for each uploaded image. The application provides an intuitive interface for users to interact with the model, making AI-powered fashion labeling accessible and easy to use.

## **Quick Start Guide**

### **Environment Setup**
Ensure you have Python 3.8+ installed. Set up your environment and install dependencies using:
```bash
pip install flask torch torchvision timm joblib pytorch-lightning
```

### **Running the Application**
To run the Flask application locally:
```bash
python app.py
```

### **Accessing the Application**
Once the application is running, you can access it via your web browser at `http://localhost:5000`.

## **Project Architecture**

Fashion Tagger is a multi-layered project, consisting of data preprocessing and model training, followed by the development of a web application that serves as the interface for the AI model. Here's a brief overview of each layer:

### **1. Data Preprocessing and Model Training**
The first layer involves data preprocessing and training the AI model. The **Fashion Product Images Dataset** is cleaned, and labels are encoded using one-hot encoding. The images are then transformed and augmented to enhance the model's learning. The model itself is an ensemble of convolutional neural networks (CNNs) including EfficientNet, ResNet, and MobileNet, which have been fine-tuned for optimal performance.

For a detailed explanation of this process, including code and experiments, please refer to the [Model Training and Preprocessing Documentation](link-to-detailed-training-readme).

### **2. Application Structure**

#### **Backend**
The backend of the Fashion Tagger application is built using Flask, a lightweight web framework for Python. It handles the following key tasks:
- **Model Inference**: Once an image is uploaded, the backend preprocesses the image and uses the trained model to generate predictions.
- **Routing and API Management**: Flask manages the routes that connect the frontend to the backend, handling user requests and returning the predicted labels.
- **File Handling**: Uploaded images are temporarily stored, and their predictions are linked for easy access and display.

#### **Frontend**
The frontend is designed to provide a user-friendly interface for interacting with the model. It includes:
- **HTML/CSS/JavaScript**: The static pages are built using HTML for structure, CSS for styling, and JavaScript for dynamic elements.
- **Responsive Design**: The frontend uses the Bulma CSS framework to ensure that the application is responsive and accessible on different devices.
- **User Interaction**: The frontend allows users to easily upload images and view the predictions in a clear and visually appealing manner.


## **Preview of Application Pages**

Here are previews of the key pages in the application:

### **1. Home Page (index.html)**
This is the landing page where users are welcomed and can start the process by clicking "Get Started."

![Home Page](path/to/first-image.png)

![About Section](path/to/second-image.png)

![Creator Info](path/to/third-image.png)

### **2. Upload Page (upload.html)**
This page allows users to upload their fashion images for prediction.

![Upload Page](path/to/fourth-image.png)

![How It Works Section](path/to/fifth-image.png)

![Project Explanation](path/to/sixth-image.png)

### **3. Result Page (result.html)**
After uploading an image, users are redirected here to see the predicted labels.

![Prediction Result](path/to/seventh-image.png)

![Detailed Labels](path/to/eighth-image.png)

> **Note**: Replace `path/to/your-image.png` with the actual path or URL where the images are stored in your repository.

## **Dataset Explanation**

The model is trained on the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). This dataset contains thousands of images labeled with categories, subcategories, colors, and other fashion-related attributes. The dataset was preprocessed by resizing images and normalizing pixel values.

## **Model Training**

The machine learning model used in this project is an ensemble of EfficientNet, ResNet, and MobileNet architectures. The training process involved several steps, including data augmentation, stratified sampling, and extensive hyperparameter tuning. For a detailed explanation of the model training, refer to the `preprocess_and_train` directory.

## **Application Workflow**

### **Backend**
The backend is built using Flask, which handles routing, file uploads, and serving the AI model for inference. The model is loaded and served using PyTorch, and predictions are made based on the uploaded images.

### **Frontend**
The frontend is designed using HTML, CSS, and JavaScript, ensuring a responsive and visually appealing user interface. The front end provides an intuitive user experience, guiding users from image upload to viewing prediction results.

## **Application Performance**

Fashion Tagger accurately predicts labels for various fashion items, demonstrating the effectiveness of the underlying AI model. Below is a summary of the application’s performance:

- **Prediction Accuracy**: High accuracy across multiple fashion categories.
- **Speed**: Real-time predictions for each uploaded image.

## **Future Work and Enhancements**

- **Model Improvement**: Further fine-tune the model to increase prediction accuracy.
- **Dataset Expansion**: Incorporate more diverse fashion datasets to improve model robustness.
- **Production Deployment**: Explore options for deploying the application using cloud services or Docker.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

- **Dataset**: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- **Frameworks**: Flask, PyTorch, PyTorch Lightning, Timm

## **Contact Information**

Created by Mahan Veisi - [LinkedIn](https://www.linkedin.com/in/mahan-veisi-427934226/) - [GitHub](https://github.com/MahanVeisi8)
