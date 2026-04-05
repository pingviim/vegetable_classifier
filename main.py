import numpy as np
import random
from config import *
from data_utils import create_dataloaders
from model import create_model
from train_utils import train_model, validate, save_metrics
from visualize import plot_training_history, plot_confusion_matrix, show_predictions



def set_seed(seed):
    """Фиксация случайных seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Создаем папку для текущего запуска
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("VEGETABLE CLASSIFICATION CNN")
    print("=" * 50)
    print(f"Run name: {RUN_NAME}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Model type: {MODEL_TYPE}")


    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")


    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


    print("\nLoading data...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(DATA_DIR)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    with open(os.path.join(SAVE_DIR, 'class_names.txt'), 'w', encoding='utf-8') as f:
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

    with open(os.path.join(SAVE_DIR, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write("CONFIGURATION\n")
        f.write("=" * 50 + "\n")
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    print("\nCreating model...")
    model = create_model(model_type=MODEL_TYPE)
    print(f"Model device: {next(model.parameters()).device}")

    print("\nTraining...")
    history, best_val_acc = train_model(
        model, train_loader, val_loader,
        EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
        DEVICE, CHECKPOINT_PATH
    )

    print("\nTesting...")
    model = create_model(model_type=MODEL_TYPE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)

    test_loss, test_acc = validate(model, test_loader, torch.nn.CrossEntropyLoss(), DEVICE)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nSaving metrics...")


    save_metrics(history, best_val_acc, test_acc, MODEL_TYPE, config_info, METRICS_PATH)



    print("\nGenerating visualizations...")
    plot_training_history(history, save_dir=SAVE_DIR)
    plot_confusion_matrix(model, test_loader, class_names, DEVICE, save_dir=SAVE_DIR)
    show_predictions(model, test_loader, class_names, DEVICE, num_images=8, save_dir=SAVE_DIR)

    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print("=" * 50)
    print(f"All results saved to: {SAVE_DIR}")
    print(f"  - Best model: {CHECKPOINT_PATH}")
    print(f"  - Metrics report: {METRICS_PATH}")
    print(f"  - Class names: {os.path.join(SAVE_DIR, 'class_names.txt')}")
    print(f"  - Configuration: {os.path.join(SAVE_DIR, 'config.txt')}")
    print(f"  - Visualizations: *.png files")


if __name__ == "__main__":
    main()