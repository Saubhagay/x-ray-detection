import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, MODEL_SAVE_PATH, CLASSES

def evaluate_model():
    print("Loading test dataset...")
    test_dir = os.path.join(DATASET_DIR, 'test')
    
    if not os.path.exists(test_dir):
        print(f"No test directory found at {test_dir}. Evaluation skipped.")
        return

    # shuffle=False is critical to align predicted labels with true labels
    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
        class_names=CLASSES
    )
    
    test_ds = test_ds.map(
        lambda x, y: (tf.keras.applications.densenet.preprocess_input(x), y), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    print("Loading trained model...")
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}. Train the model first.")
        return
        
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    print("Evaluating model...")
    results = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC: {results[2]:.4f}")
    
    print("Generating predictions...")
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'confusion_matrix.png'))
    print("Saved confusion matrix to results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
