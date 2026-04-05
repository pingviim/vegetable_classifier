import torch
import torch.nn as nn
from tqdm import tqdm
import os

def save_metrics(history, best_val_acc, test_acc, model_type, config_info, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("VEGETABLE CLASSIFICATION - TRAINING METRICS\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("TRAINING HISTORY BY EPOCH\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} {'Val Loss':<14} {'Val Acc':<12} {'Best':<8}\n")
        f.write("-" * 80 + "\n")

        best_val_acc_epoch = 0
        best_epoch = 0

        for epoch in range(len(history['train_loss'])):
            current_val_acc = history['val_acc'][epoch]
            is_best = "BEST" if current_val_acc > best_val_acc_epoch else ""
            if current_val_acc > best_val_acc_epoch:
                best_val_acc_epoch = current_val_acc
                best_epoch = epoch + 1

            f.write(f"{epoch + 1:<8} "
                    f"{history['train_loss'][epoch]:<14.6f} "
                    f"{history['train_acc'][epoch]:<12.2f} "
                    f"{history['val_loss'][epoch]:<14.6f} "
                    f"{history['val_acc'][epoch]:<12.2f} "
                    f"{is_best:<8}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("BEST RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc_epoch:.2f}%\n")

    print(f"Metrics saved to {save_path}")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, device, save_path):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (Acc: {val_acc:.2f}%)")

        scheduler.step()

    return history, best_val_acc