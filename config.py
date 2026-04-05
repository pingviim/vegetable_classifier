import torch
from datetime import datetime
import os

### Пути до датасета и места сохранения запусков
DATA_DIR = "./dataset"
BASE_SAVE_DIR = "./runs"

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(BASE_SAVE_DIR, RUN_NAME)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "best_model.pth")
METRICS_PATH = os.path.join(SAVE_DIR, "metrics.txt")

### Параметры для обучения
IMG_SIZE = (224, 224)
NUM_CLASSES = 15
USE_AUGMENTATION = True
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### Тип модели - custom или resnet50
MODEL_TYPE = 'resnet50'

SEED = 42