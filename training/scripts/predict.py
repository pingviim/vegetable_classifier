import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from config import DEVICE, IMG_SIZE, BASE_SAVE_DIR
from training.utils.model_utils import create_model


def load_model_for_prediction(model_path, model_type='custom', device=DEVICE):
    """Загрузка обученной модели для предсказаний"""
    model = create_model(model_type=model_type, device=device)
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


def find_latest_model(model_type='resnet50'):
    """Находит последнюю обученную модель в структуре training/models/{model_type}/"""
    model_dir = os.path.join(BASE_SAVE_DIR, model_type)

    if not os.path.exists(model_dir):
        return None, None, None

    model_path = os.path.join(model_dir, "best_model.pth")
    class_names_path = os.path.join(model_dir, "class_names.txt")
    config_path = os.path.join(model_dir, "config.txt")

    if not os.path.exists(model_path) or not os.path.exists(class_names_path):
        return None, None, None

    return model_path, class_names_path, config_path


# Использование
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict vegetable class from image')
    parser.add_argument('--model_type', type=str, choices=['resnet50', 'custom'],
                        default='resnet50', help='Model type to use for prediction')
    parser.add_argument('--image_path', type=str, help='Path to image file')
    parser.add_argument('--model_path', type=str, help='Path to specific model file')

    args = parser.parse_args()

    # Поиск модели
    if args.model_path:
        model_path = args.model_path
        model_dir = os.path.dirname(model_path)
        class_names_path = os.path.join(model_dir, "class_names.txt")
        config_path = os.path.join(model_dir, "config.txt")
        model_type = args.model_type
    else:
        model_path, class_names_path, config_path = find_latest_model(args.model_type)
        if model_path is None:
            print(f"No model found for type: {args.model_type}")
            print(f"Expected path: {BASE_SAVE_DIR}/{args.model_type}/best_model.pth")
            exit(1)
        model_type = args.model_type

    # Загружаем классы
    if not os.path.exists(class_names_path):
        print(f"Class names file not found: {class_names_path}")
        exit(1)

    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Определяем тип модели из конфига (если есть)
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'model_type' in line:
                    model_type = line.split(':')[1].strip()
                    break

    print(f"=" * 50)
    print(f"Loading model from: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Classes: {class_names}")
    print(f"=" * 50)

    # Загрузка модели
    model = load_model_for_prediction(model_path, model_type=model_type)

    # Предсказание
    if args.image_path:
        img_path = args.image_path
    else:
        img_path = input("Enter image path: ").strip()

    if os.path.exists(img_path):
        pred_class, confidence = predict_image(img_path, model, class_names)
        print(f"\nPredicted: {pred_class} ({confidence:.2%})")
    else:
        print(f"Image not found: {img_path}")