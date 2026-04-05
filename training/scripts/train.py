import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import random
import torch
from datetime import datetime
from config import *
from training.utils.data_utils import create_dataloaders
from training.utils.model_utils import create_model
from training.utils.train_utils import train_model, validate, save_metrics
from training.utils.visualize import plot_training_history, plot_confusion_matrix, show_predictions

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(BASE_SAVE_DIR, MODEL_TYPE)
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    metrics_path = os.path.join(save_dir, "metrics.txt")

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("VEGETABLE CLASSIFICATION TRAINING")
    print("=" * 50)
    print(f"Run name: {run_name}")
    print(f"Save directory: {save_dir}")
    print(f"Model type: {MODEL_TYPE}")

    set_seed(SEED)
    print(f"Device: {DEVICE}")

    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(DATA_DIR)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    with open(os.path.join(save_dir, 'class_names.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))

    config_info = {
        'model_type': MODEL_TYPE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'dropout_rate': DROPOUT_RATE,
        'use_augmentation': USE_AUGMENTATION,
        'img_size': IMG_SIZE,
        'seed': SEED,
        'device': DEVICE
    }

    with open(os.path.join(save_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write("CONFIGURATION\n")
        f.write("=" * 50 + "\n")
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    print("\nCreating model...")
    model = create_model(model_type=MODEL_TYPE)

    print("\nTraining...")
    history, best_val_acc = train_model(
        model, train_loader, val_loader,
        EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
        DEVICE, checkpoint_path
    )

    print("\nTesting...")
    model = create_model(model_type=MODEL_TYPE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)

    test_loss, test_acc = validate(model, test_loader, torch.nn.CrossEntropyLoss(), DEVICE)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nSaving metrics...")
    save_metrics(history, best_val_acc, test_acc, MODEL_TYPE, config_info, metrics_path)

    print("\nGenerating visualizations...")
    plot_training_history(history, save_dir=save_dir)
    plot_confusion_matrix(model, test_loader, class_names, DEVICE, save_dir=save_dir)
    show_predictions(model, test_loader, class_names, DEVICE, num_images=8, save_dir=save_dir)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"All results saved to: {save_dir}")

if __name__ == "__main__":
    main()