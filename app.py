from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Class labels matching your training
CARIES_CLASSES = ['healthy', 'incipient_caries', 'moderate_caries', 'deep_caries']
IMG_SIZE = 256

# Load model
try:
    model = tf.keras.models.load_model(
        'trained_models/cariex_classifier/best_model.keras'
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
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(f"Received file: {file.filename}")

    try:
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]  # (4,)
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CARIES_CLASSES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Build result
        result = {
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'has_caries': predicted_class != 'healthy',
            'all_probabilities': {
                CARIES_CLASSES[i]: round(float(predictions[i]) * 100, 2)
                for i in range(len(CARIES_CLASSES))
            }
        }

        print(f"Result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)