import os

# Limit TensorFlow memory usage before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN to save memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU only

import base64
import gc
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import IMG_HEIGHT, IMG_WIDTH, CLASSES, MODEL_SAVE_PATH

# Optimize TF threads to minimize memory footprint
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
# Allow CORS for all domains, specifically helpful since frontend will be deployed separately
CORS(app)

# Load the AI model globally when the server starts
print("Loading model for the web app...")
if os.path.exists(MODEL_SAVE_PATH):
    # Load with compile=False to save memory (we don't need the optimizer for inference)
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    print("Model loaded.")
else:
    model = None
    print(f"WARNING: Model not found at {MODEL_SAVE_PATH}. Please train first.")

# Heatmap functionality disabled for Free Tier memory limits
def make_gradcam_heatmap(*args, **kwargs):
    pass

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running. Please use the frontend UI to interact with this backend."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

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
            
            # Preprocess
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = tf.keras.applications.densenet.preprocess_input(img_array.copy())
            
            # Predict using lightweight __call__ instead of heavy .predict()
            preds_tensor = model(preprocessed_img, training=False)
            preds = preds_tensor.numpy()
            
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
