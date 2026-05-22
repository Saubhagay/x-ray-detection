import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

def build_model(fine_tune=False):
    """
    Builds the DenseNet121 transfer learning model.
    """
    # Load DenseNet121 without the top classification layer
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model by default for transfer learning
    base_model.trainable = fine_tune
    
    if fine_tune:
        # If fine-tuning, unfreeze the last few blocks of the model
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
    # Add custom dense layers for our 3-class classification
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # High dropout to prevent overfitting
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax') # Softmax for 3 classes
    ])
    
    return model

if __name__ == "__main__":
    # Test model building
    model = build_model()
    model.summary()
