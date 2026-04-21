# Handwritten Digit Recognition System

A deep learning-based handwritten digit recognition system built with PyTorch and Flask, featuring a three-layer CNN architecture that achieves over 98% accuracy on the MNIST dataset.

## Features

- **Deep Learning Model**: Three-layer CNN (32→64→128 channels) with max pooling, dropout regularization, and fully connected layers
- **Smart Image Preprocessing**: Automatic centering, size normalization, and MNIST standardization for consistent inference
- **Real-time Inference**: RESTful API supporting Base64 encoded images with prediction results and confidence scores
- **Multiple UI Themes**: Four different frontend interfaces with responsive design and touch support
- **One-click Setup**: Automated dependency checking, model training, and service startup

## Project Structure

```
digital-recognition/
├── app.py                  # Flask backend application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── data/                   # Data directory
│   └── .gitkeep           # Keep directory in git
└── templates/              # HTML templates
    ├── index.html         # Default blue theme
    ├── index_new.html     # Modern gradient purple theme
    ├── view1.html         # Dark sci-fi style
    └── view2.html         # Retro green terminal style
```

## Quick Start

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- Flask 2.0+

### Installation

1. **Create virtual environment (recommended)**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train_model.py
```
The MNIST dataset (~100MB) will be downloaded automatically. The trained model is saved to `./data/digit_recognition_cnn.pth`.

4. **Start the web application**
```bash
python app.py
```

5. **Access the interface**

Open your browser and navigate to:
- `http://localhost:5000` - Default interface
- `http://localhost:5000/new` - Modern interface (recommended)
- `http://localhost:5000/view1` - Sci-fi style
- `http://localhost:5000/view2` - Retro terminal style

## Technical Details

### Tech Stack

- **Deep Learning**: PyTorch 1.9+
- **Web Framework**: Flask 2.0+
- **Image Processing**: Pillow, NumPy
- **Data Transformation**: torchvision.transforms

### Model Architecture

```
Input (1×28×28)
  ↓
Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)  [14×14]
  ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)  [7×7]
  ↓
Conv2d(64→128, 3×3) + ReLU + MaxPool(2×2)  [3×3]
  ↓
Flatten(1152) → FC(128) + ReLU + Dropout(0.5)
  ↓
FC(10) → LogSoftmax
```

**Training Configuration**:
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Epochs: 15
- Data Augmentation: Random rotation (±10°), random translation (±10%)

### Image Preprocessing Pipeline

1. Base64 decode → PIL grayscale conversion
2. Bilinear interpolation resize to 28×28
3. Threshold segmentation (>50) to extract digit bounding box
4. Crop and center placement on 28×28 canvas
5. Convert to Tensor and normalize to [0,1]
6. MNIST normalization: `(x - 0.1307) / 0.3081`

### API Endpoint

**POST /predict**

Request:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

Response:
```json
{
  "prediction": 7,
  "confidence": 96.85,
  "probabilities": [0.001, 0.002, ..., 0.968, ...]
}
```

## Usage

1. Navigate to any interface (recommend `/new`)
2. Draw a digit (0-9) on the canvas using mouse or touch
3. Click "Recognize" button
4. View prediction result, confidence score, and probability distribution
5. Click "Clear" to redraw

## Key Modules

### app.py

- **Net class**: CNN network architecture (consistent with training)
- **preprocess_image()**: Image preprocessing with centering and normalization
- **Routes**: Multiple UI endpoints and prediction API

### train_model.py

- **Data Loading**: Automatic MNIST download with augmentation
- **Net class**: Three-layer CNN model definition
- **train()/test()**: Training and evaluation functions
- **Auto-save**: Saves best model weights based on accuracy

## Dependencies

```txt
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
numpy>=1.21.0
pillow>=8.3.0
matplotlib>=3.4.0
```

## Performance

- **Test Accuracy**: >98%
- **Inference Time**: <50ms per image (CPU)
- **Model Size**: ~5MB
- **Input Support**: Mouse and touch screen

## Notes

1. First run requires downloading MNIST dataset (~100MB)
2. Training time varies by hardware, typically 5-15 minutes
3. Draw digits centered and filling the canvas for best results
4. Ensure `./data` directory has write permissions
5. Recommended browsers: Chrome, Firefox, Edge

## Troubleshooting

**Q: Model file not found?**  
A: Run `python train_model.py` to train the model, or check `data/` directory permissions

**Q: Low recognition accuracy?**  
A: Ensure clear drawing, centered digit, and adequate stroke thickness

**Q: Port already in use?**  
A: Modify the port parameter in `app.py` line 167

## Future Enhancements

- Add model visualization (TensorBoard)
- Support batch image upload
- Integrate additional datasets (e.g., EMNIST)
- Deploy as Docker container
- Add user authentication and history tracking

## License

This project is for educational and research purposes only.

---

**Note**: This project demonstrates a complete deep learning application workflow including data preparation, model design, training optimization, and web deployment.
