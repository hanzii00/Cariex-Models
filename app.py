from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define custom objects
def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou_score(y_true, y_pred):
    smooth = 1.
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

# Load model
try:
    model = tf.keras.models.load_model(
        'trained_models/adult_tooth_unet/best_model.h5',
        custom_objects={
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'iou_score': iou_score
        },
        compile=False
    )
    print("✓ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model = None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        print("No image in request")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    print(f"Received file: {file.filename}")
    
    try:
        # Read image
        img_bytes = file.read()
        print(f"Image size: {len(img_bytes)} bytes")
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        print(f"Original image size: {img.size}")
        
        # Resize to model input
        img = img.resize((256, 256))
        print(f"Resized to: {img.size}")
        
        img_array = np.array(img) / 255.0
        print(f"Array shape: {img_array.shape}")
        
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Batch shape: {img_array.shape}")
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array, verbose=0)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction range: {prediction.min()} to {prediction.max()}")
        
        # Calculate confidence
        affected_pixels = np.sum(prediction > 0.5)
        total_pixels = prediction.size
        confidence = float(affected_pixels / total_pixels)
        
        result = {
            'hasCaries': bool(confidence > 0.1),
            'confidence': min(confidence * 10, 1.0)
        }
        
        print(f"Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)