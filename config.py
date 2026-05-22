import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Class names
CLASSES = ['NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
NUM_CLASSES = len(CLASSES)

# Model paths
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'densenet121_xray.keras')
