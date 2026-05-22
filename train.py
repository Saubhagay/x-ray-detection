import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import build_model
from config import DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, CLASSES

def get_datasets():
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Dataset directory not found at {train_dir}. Please organize your dataset folders.")

    # Data Augmentation to prevent overfitting
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        class_names=CLASSES
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        class_names=CLASSES
    )
    
    # Preprocess inputs specifically for DenseNet
    def preprocess(image, label, augment=False):
        if augment:
            image = data_augmentation(image, training=True)
        # DenseNet requires specific preprocessing (-1 to 1 or 0 to 1 scaling based on weights)
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label

    train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def train():
    print("Loading datasets...")
    train_ds, val_ds = get_datasets()
    
    print("Building model...")
    model = build_model(fine_tune=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"Training completed. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
