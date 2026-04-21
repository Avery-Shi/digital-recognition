from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import base64
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Define CNN model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

# Initialize model and load weights
model = Net()
model_path = './data/digit_recognition_cnn.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"Warning: Model file {model_path} not found. Please run training script first.")


def preprocess_image(image_data):
    """
    Preprocess image data for model inference
    """
    # Decode base64 image to PIL grayscale image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Resize to MNIST standard size (28x28)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Center the digit in the image
    coords = np.where(image_array > 50)
    if coords[0].size > 0 and coords[1].size > 0:
        top, left = np.min(coords[0]), np.min(coords[1])
        bottom, right = np.max(coords[0]), np.max(coords[1])
        
        cropped = image_array[top:bottom+1, left:right+1]
        crop_height, crop_width = cropped.shape
        center_x = (28 - crop_width) // 2
        center_y = (28 - crop_height) // 2
        
        centered_image = np.zeros((28, 28), dtype=np.uint8)
        centered_image[center_y:center_y+crop_height, center_x:center_x+crop_width] = cropped
    else:
        centered_image = image_array
    
    # Convert to tensor and normalize
    image_tensor = torch.tensor(centered_image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    
    # Apply MNIST normalization
    mean = 0.1307
    std = 0.3081
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor
# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view1')
def view1():
    return render_template('view1.html')

@app.route('/view2')
def view2():
    return render_template('view2.html')

@app.route('/new')
def index_new():
    return render_template('index_new.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        image_data = request.json['image']
        processed_image = preprocess_image(image_data)
        
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.exp(output)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
            all_probabilities = probabilities.squeeze().tolist()
        
        result = {
            'prediction': int(predicted_class),
            'confidence': round(confidence * 100, 2),
            'probabilities': all_probabilities
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)