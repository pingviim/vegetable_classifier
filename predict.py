# predict.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from config import DEVICE, IMG_SIZE
from model import create_model


def load_model_for_prediction(model_path, model_type='custom', device=DEVICE):
    """Загрузка обученной модели для предсказаний"""
    model = create_model(model_type=model_type)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_image(image_path, model, class_names, device=DEVICE):
    """Предсказание одного изображения"""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, 1)

    predicted_class = class_names[top_class.item()]
    confidence = top_prob.item()

    # Визуализация
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.barh([predicted_class], [confidence], color='green')
    plt.xlim(0, 1)
    plt.xlabel('Confidence')
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}')

    plt.tight_layout()
    plt.show()

    return predicted_class, confidence


def predict_batch(image_paths, model, class_names, device=DEVICE):
    """Предсказание для нескольких изображений"""
    results = []
    for img_path in image_paths:
        pred_class, confidence = predict_image(img_path, model, class_names, device)
        results.append({
            'image': img_path,
            'predicted_class': pred_class,
            'confidence': confidence
        })
    return results


# Использование
if __name__ == "__main__":
    # Автоматически находим последний запуск
    runs_dir = "./runs"
    if not os.path.exists(runs_dir):
        print("No runs found! Please train a model first.")
        exit()

    latest_run = sorted(os.listdir(runs_dir))[-1]  # берем последний запуск
    model_path = os.path.join(runs_dir, latest_run, "best_model.pth")
    class_names_path = os.path.join(runs_dir, latest_run, "class_names.txt")
    config_path = os.path.join(runs_dir, latest_run, "config.txt")

    # Загружаем классы
    if not os.path.exists(class_names_path):
        print(f"Class names file not found: {class_names_path}")
        exit()

    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Определяем тип модели из конфига
    model_type = 'custom'  # по умолчанию
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'model_type' in line:
                    model_type = line.split(':')[1].strip()
                    break

    print(f"Loading model from: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Classes: {class_names}")

    # Загрузка модели
    model = load_model_for_prediction(model_path, model_type=model_type)

    # Предсказание
    img_path = input("Enter image path: ").strip()
    if os.path.exists(img_path):
        pred_class, confidence = predict_image(img_path, model, class_names)
        print(f"\nPredicted: {pred_class} ({confidence:.2%})")
    else:
        print(f"Image not found: {img_path}")