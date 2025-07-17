import os
import ssl
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

# Disable SSL verification for VGG19 weights download
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('vgg_unfrozen.weights.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Brain Tumor Detected"
    else:
        return "Unknown"

def preprocess_image(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    # Resize to match model's expected sizing
    img = cv2.resize(img, (240, 240))
    
    # Convert BGR to RGB (OpenCV loads as BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions to create a batch of 1
    img = np.expand_dims(img, axis=0)
    
    # Preprocess the image using VGG19's preprocess_input
    img = preprocess_input(img)
    
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            processed_img = preprocess_image(filepath)
            
            # Make prediction
            predictions = model_03.predict(processed_img)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            result = {
                'class': int(predicted_class),
                'label': get_className(predicted_class),
                'confidence': round(confidence * 100, 2)
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed. Please upload an image file (PNG, JPG, JPEG).'}), 400

if __name__ == '__main__':
    app.run(debug=True)
