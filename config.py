import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "training", "dataset")
BASE_SAVE_DIR = os.path.join(PROJECT_ROOT, "training", "models")
RUNS_DIR = os.path.join(PROJECT_ROOT, "training", "runs")

IMG_SIZE = (224, 224)
NUM_CLASSES = 15
USE_AUGMENTATION = True
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_TYPE = 'custom'

SEED = 42

API_MODELS_PATH = os.path.join(PROJECT_ROOT, "training", "models")