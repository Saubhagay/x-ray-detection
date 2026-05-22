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
    
    # Pre-build Grad-CAM model globally to prevent memory leaks during prediction
    print("Building Grad-CAM model...")
    base_model = model.layers[0]
    grad_model = tf.keras.models.Model(
        base_model.inputs, [base_model.get_layer('relu').output, base_model.output]
    )
else:
    model = None
    print(f"WARNING: Model not found at {MODEL_SAVE_PATH}. Please train first.")

def make_gradcam_heatmap(img_array, model, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, base_preds = grad_model(img_array)
        x = base_preds
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

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
            
            # Predict
            preds = model.predict(preprocessed_img)
            pred_class_idx = np.argmax(preds[0])
            pred_class = CLASSES[pred_class_idx]
            confidence = float(preds[0][pred_class_idx] * 100)
            
            # All probabilities for chart
            probs = {CLASSES[i]: float(preds[0][i]*100) for i in range(len(CLASSES))}

            # Grad-CAM
            heatmap = make_gradcam_heatmap(preprocessed_img, model, grad_model, pred_class_idx)
            
            heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            superimposed_img = heatmap_colormap * 0.4 + img_resized
            superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR) # For encoding
            
            # Convert Original and Heatmap to Base64 to return directly (Memory efficiency, no disk usage!)
            _, orig_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            orig_base64 = base64.b64encode(orig_encoded).decode('utf-8')
            
            _, heatmap_encoded = cv2.imencode('.jpg', superimposed_img)
            heatmap_base64 = base64.b64encode(heatmap_encoded).decode('utf-8')

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
