import os
import base64
import gc
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import IMG_HEIGHT, IMG_WIDTH, CLASSES

# Limit TensorFlow memory usage before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow.lite as tflite

app = Flask(__name__)
# Allow CORS for all domains
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "model.tflite")

# Load the lightweight TFLite model globally
print("Loading TFLite model...")
if os.path.exists(TFLITE_MODEL_PATH):
    try:
        # Read the file directly into memory to bypass Linux path/C++ wrapper bugs
        with open(TFLITE_MODEL_PATH, 'rb') as f:
            model_content = f.read()
        interpreter = tflite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite Model loaded successfully.")
    except Exception as e:
        interpreter = None
        print(f"CRITICAL ERROR loading TFLite model: {e}")
else:
    interpreter = None
    print(f"WARNING: Model not found at {TFLITE_MODEL_PATH}. Please run convert_to_tflite.py first.")

# Heatmap functionality disabled for Free Tier memory limits
def make_gradcam_heatmap(*args, **kwargs):
    pass

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running. Please use the frontend UI to interact with this backend."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Model not loaded. Please convert the model to TFLite first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            # Read image directly from memory
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Preprocess manually without TensorFlow
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
            
            img_array = img_resized.astype(np.float32)
            img_array /= 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            preprocessed_img = (img_array - mean) / std
            preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
            
            # Predict using ultra-lightweight TFLite
            interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])
            
            pred_class_idx = np.argmax(preds[0])
            pred_class = CLASSES[pred_class_idx]
            confidence = float(preds[0][pred_class_idx] * 100)
            
            # All probabilities for chart
            probs = {CLASSES[i]: float(preds[0][i]*100) for i in range(len(CLASSES))}

            # Convert Original to Base64
            _, orig_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            orig_base64 = base64.b64encode(orig_encoded).decode('utf-8')
            
            # Disable Heatmap to save memory. Use original image as placeholder.
            heatmap_base64 = orig_base64

            return jsonify({
                'success': True,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': probs,
                'original_image': f"data:image/jpeg;base64,{orig_base64}",
                'heatmap_image': f"data:image/jpeg;base64,{heatmap_base64}"
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Force garbage collection after prediction to free up RAM
            gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
