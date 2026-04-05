import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm import tqdm

### Отрисовка графиков
def plot_training_history(history, save_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_dir}")

    plt.show()


def plot_confusion_matrix(model, loader, class_names, device, save_dir=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Creating confusion matrix'):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_dir}")

    plt.show()

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    if save_dir:
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        print(f"✓ Classification report saved to {save_dir}")


def show_predictions(model, loader, class_names, device, num_images=8, save_dir=None):
    model.eval()
    images, labels = next(iter(loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, 1)

    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.ravel()

    for idx in range(num_images):
        img = images[idx].cpu().permute(1, 2, 0)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        axes[idx].imshow(img)
        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        conf = probs[idx, preds[idx]].item()

        color = 'green' if preds[idx] == labels[idx] else 'red'
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label} ({conf:.2f})', color=color)
        axes[idx].axis('off')

    plt.tight_layout()

    # Сохраняем если указана папка
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Sample predictions saved to {save_dir}")

    plt.show()