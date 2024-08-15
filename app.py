from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
import torch
import numpy as np
from PIL import Image
import joblib
from torchvision import transforms
import pytorch_lightning as pl
import timm
import os
import torch.nn as nn

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages

# Define your FashionModel class with the ensemble structure
class FashionModel(pl.LightningModule):
    def __init__(self, num_classes, none_label_mask, label_dicts, freeze_backbone=True):
        super(FashionModel, self).__init__()
        # Define base models without pre-trained weights
        self.model_efficientnet = timm.create_model('efficientnet_b3', pretrained=False)
        self.model_resnet = timm.create_model('resnet50', pretrained=False)
        self.model_mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=False)

        # Extract the number of features from each model's classifier
        self.efficientnet_features = self.model_efficientnet.classifier.in_features
        self.resnet_features = self.model_resnet.fc.in_features
        self.mobilenet_features = self.model_mobilenet.classifier.in_features

        # Replace the classifiers with identity to keep features
        self.model_efficientnet.classifier = nn.Identity()
        self.model_resnet.fc = nn.Identity()
        self.model_mobilenet.classifier = nn.Identity()

        # Define the ensemble fully connected layers
        total_features = self.efficientnet_features + self.resnet_features + self.mobilenet_features
        self.ensemble_fc = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.none_label_mask = none_label_mask
        self.label_dicts = label_dicts
        self.num_classes = num_classes

    def forward(self, x):
        features_efficientnet = self.model_efficientnet(x)
        features_resnet = self.model_resnet(x)
        features_mobilenet = self.model_mobilenet(x)

        combined_features = torch.cat([features_efficientnet, features_resnet, features_mobilenet], dim=1)
        return self.ensemble_fc(combined_features)

# Load label dictionaries and mask
label_dicts = joblib.load('instance/label_dicts.pkl')
none_label_mask = joblib.load('instance/none_label_mask.pkl')

# Define the path to the model checkpoint
checkpoint_path = 'best_model-epoch=35-val_loss=3.75.ckpt'

# Initialize the model
num_classes = len(label_dicts['gender']) + len(label_dicts['masterCategory']) + len(label_dicts['subCategory']) + len(label_dicts['articleType']) + len(label_dicts['baseColour']) + len(label_dicts['season']) + len(label_dicts['usage'])
model = FashionModel(num_classes, none_label_mask, label_dicts)

# Load the model weights from the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model to evaluation mode
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Decode predictions
def decode_predictions(predictions, label_dicts):
    decoded_labels = []
    for pred in predictions:
        sample_labels = {}
        start_idx = 0
        for col, categories in label_dicts.items():
            end_idx = start_idx + len(categories)
            sample_pred = pred[start_idx:end_idx]
            
            # Sort by probability
            sorted_indices = np.argsort(sample_pred)[::-1]
            top1_idx = sorted_indices[0]
            top1_label = categories[top1_idx]
            top1_prob = sample_pred[top1_idx]
            
            # Find the next top prediction that is not 'None'
            next_top_idx = None
            for idx in sorted_indices[1:]:
                if categories[idx] != 'None':
                    next_top_idx = idx
                    break
            
            if next_top_idx is not None:
                top2_label = categories[next_top_idx]
                top2_prob = sample_pred[next_top_idx]
            else:
                top2_label = None
                top2_prob = None
            
            # Only add top1_label if it is not 'None'
            if top1_label != 'None':
                if top1_prob >= 0.5:
                    sample_labels[col] = (top1_label, top1_prob)
                else:
                    sample_labels[col] = (top1_label, top1_prob)
                    if top2_label is not None:
                        sample_labels[f'{col}_second'] = (top2_label, top2_prob)
            else:
                if top2_label is not None:
                    sample_labels[col] = (top2_label, top2_prob)
            
            start_idx = end_idx
        decoded_labels.append(sample_labels)
    return decoded_labels

# Function to preprocess and predict a single image
def predict_image(image_path, model, label_dicts, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))  # Resize image to fit within 256x256
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.sigmoid(output).cpu().numpy()[0]
    decoded_prediction = decode_predictions([probabilities], label_dicts)[0]
    return decoded_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            prediction = predict_image(file_path, model, label_dicts, transform, device)
            return redirect(url_for('result', filename=filename, prediction=prediction))
        else:
            flash('File type not allowed')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/result')
def result():
    filename = request.args.get('filename')
    prediction = eval(request.args.get('prediction'))  # Convert string to dictionary
    return render_template('result.html', prediction=prediction, image_path=filename)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('./uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    app.run(debug=True)
