# Brain Tumor Classification Web Application

A web-based application for classifying brain tumor MRI scans using deep learning. This application uses a fine-tuned VGG19 model to classify brain MRI images into different tumor categories.

## Features

- **User-friendly Web Interface**: Simple and intuitive interface for uploading MRI scans
- **Deep Learning Model**: Utilizes a pre-trained VGG19 model fine-tuned for brain tumor classification
- **Real-time Prediction**: Get instant classification results with confidence scores
- **Responsive Design**: Works on both desktop and mobile devices

## Demo

![Demo](static/uploads/Brain_Tumor_Classification_Image.jpg)

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- macOS (for M1/M2 chip support with tensorflow-metal)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   Open your web browser and go to `http://localhost:5000`

3. **Upload an MRI scan**
   - Click the upload button to select an MRI scan image
   - The application will process the image and display the prediction

## Project Structure

```
brain-tumor-classification/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── vgg_unfrozen.weights.h5  # Pre-trained model weights
├── static/               # Static files (CSS, JS, images)
│   └── uploads/          # Directory for uploaded images
└── templates/            # HTML templates
    └── index.html        # Main web interface
```

## Model Details

The model is based on VGG19 architecture, pre-trained on ImageNet and fine-tuned on a brain tumor MRI dataset. It can classify MRI scans into four categories:

- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

## API Endpoints

- `GET /`: Main application interface
- `POST /upload`: Endpoint for uploading and processing MRI images

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [VGG19](https://arxiv.org/abs/1409.1556) - The base model architecture
- [TensorFlow](https://www.tensorflow.org/) - Machine learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework

## Contact

For any questions or feedback, please open an issue on GitHub.
