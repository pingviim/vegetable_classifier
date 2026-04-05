import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from gradcam import (
    visualize_single_image,
    visualize_top_classes,
    analyze_model_decisions,
    save_gradcam_examples,
    GradCAM
)
from config import DEVICE
from model import create_model

CLASS_NAMES = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
               'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

MODEL_TYPE = "custom"


def get_test_loader():
    test_dir = './../dataset/test'

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Папка {test_dir} не найдена. Запустите download_dataset_for_test.py")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"Загружено {len(dataset)} изображений, {len(dataset.classes)} классов")
    return loader, dataset


def analyze_heatmaps_by_class(model, model_type, dataloader, class_names, device, num_per_class=5):
    print("\n" + "=" * 60)
    print("АНАЛИЗ HEATMAP ПО КЛАССАМ")
    print("=" * 60)

    if model_type == 'custom':
        layer = model.conv4
    else:
        layer = model.backbone.layer4

    gradcam = GradCAM(model, layer)
    model.eval()

    class_heatmaps = {name: [] for name in class_names}
    class_attention_stats = {name: {'center_focus': 0, 'edge_focus': 0, 'texture_focus': 0} for name in class_names}

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        for j in range(len(images)):
            label = labels[j].item()
            class_name = class_names[label]

            if len(class_heatmaps[class_name]) >= num_per_class:
                continue

            heatmap, _, confidence = gradcam.generate(images[j:j + 1], class_idx=label)

            # Анализируем heatmap
            focus_type = analyze_heatmap_focus(heatmap)
            class_attention_stats[class_name][focus_type] += 1

            class_heatmaps[class_name].append({
                'heatmap': heatmap,
                'confidence': confidence,
                'image': images[j].cpu()
            })

        if all(len(heatmaps) >= num_per_class for heatmaps in class_heatmaps.values()):
            break

    visualize_class_heatmaps(class_heatmaps, class_names, num_per_class)

    print("\n" + "=" * 60)
    print("СТАТИСТИКА ВНИМАНИЯ МОДЕЛИ ПО КЛАССАМ")
    print("=" * 60)
    print(f"{'Класс':<20} {'Центр':<10} {'Края':<10} {'Текстура':<10}")
    print("-" * 50)

    for class_name in class_names:
        stats = class_attention_stats[class_name]
        total = sum(stats.values())
        if total > 0:
            center_pct = stats['center_focus'] / total * 100
            edge_pct = stats['edge_focus'] / total * 100
            texture_pct = stats['texture_focus'] / total * 100
            print(f"{class_name:<20} {center_pct:>6.1f}%    {edge_pct:>6.1f}%    {texture_pct:>6.1f}%")

    return class_heatmaps, class_attention_stats


def analyze_heatmap_focus(heatmap, center_radius=0.3):
    h, w = heatmap.shape
    center_h, center_w = h // 2, w // 2
    radius_h, radius_w = int(h * center_radius), int(w * center_radius)

    center_mask = np.zeros_like(heatmap)
    center_mask[center_h - radius_h:center_h + radius_h,
    center_w - radius_w:center_w + radius_w] = 1

    edge_mask = np.ones_like(heatmap)
    edge_mask[center_h - radius_h:center_h + radius_h,
    center_w - radius_w:center_w + radius_w] = 0

    center_activation = (heatmap * center_mask).sum() / (center_mask.sum() + 1e-8)
    edge_activation = (heatmap * edge_mask).sum() / (edge_mask.sum() + 1e-8)

    if center_activation > edge_activation * 1.5:
        return 'center_focus'
    elif edge_activation > center_activation * 1.5:
        return 'edge_focus'
    else:
        return 'texture_focus'


def visualize_class_heatmaps(class_heatmaps, class_names, num_examples=5):
    classes_per_group = 5
    n_classes = len(class_names)
    n_groups = (n_classes + classes_per_group - 1) // classes_per_group

    for group_idx in range(n_groups):
        start_idx = group_idx * classes_per_group
        end_idx = min(start_idx + classes_per_group, n_classes)
        group_classes = class_names[start_idx:end_idx]
        n_classes_in_group = len(group_classes)

        fig, axes = plt.subplots(n_classes_in_group, num_examples + 1,
                                 figsize=(3 * (num_examples + 1), 3 * n_classes_in_group))

        if n_classes_in_group == 1:
            axes = axes.reshape(1, -1)

        for i, class_name in enumerate(group_classes):
            axes[i, 0].text(0.5, 0.5, class_name, ha='center', va='center',
                            fontsize=10, fontweight='bold', wrap=True)
            axes[i, 0].axis('off')

            heatmaps = class_heatmaps[class_name]
            for j in range(min(num_examples, len(heatmaps))):
                heatmap = heatmaps[j]['heatmap']
                confidence = heatmaps[j]['confidence']

                im = axes[i, j + 1].imshow(heatmap, cmap='jet')
                axes[i, j + 1].set_title(f'{confidence:.2f}', fontsize=8)
                axes[i, j + 1].axis('off')

                if group_idx == 0 and i == 0 and j == 0:
                    plt.colorbar(im, ax=axes[i, j + 1], fraction=0.046)

        plt.suptitle(f'Grad-CAM Heatmaps by Class (Group {group_idx + 1}/{n_groups})',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'results/heatmaps_by_class_group_{group_idx + 1}.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Сохранена группа {group_idx + 1}/{n_groups}: классы {start_idx + 1}-{end_idx}")


def interpret_model_decisions(model, model_type, dataloader, class_names, device, num_samples=20):
    if model_type == 'custom':
        layer = model.conv4
    else:
        layer = model.backbone.layer4

    gradcam = GradCAM(model, layer)
    model.eval()

    correct_predictions = []
    wrong_predictions = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        for j in range(len(images)):
            if len(correct_predictions) >= num_samples // 2 and len(wrong_predictions) >= num_samples // 2:
                break

            with torch.no_grad():
                output = model(images[j:j + 1])
                pred = output.argmax(dim=1).item()
                true = labels[j].item()
                confidence = torch.softmax(output, dim=1)[0, pred].item()

            sample_info = {
                'image': images[j],
                'true_class': class_names[true],
                'pred_class': class_names[pred],
                'confidence': confidence,
                'correct': pred == true
            }

            if pred == true and len(correct_predictions) < num_samples // 2:

                heatmap, _, _ = gradcam.generate(images[j:j + 1], class_idx=pred)
                sample_info['heatmap'] = heatmap
                correct_predictions.append(sample_info)
            elif pred != true and len(wrong_predictions) < num_samples // 2:
                heatmap, _, _ = gradcam.generate(images[j:j + 1], class_idx=pred)
                sample_info['heatmap'] = heatmap
                wrong_predictions.append(sample_info)

        if len(correct_predictions) >= num_samples // 2 and len(wrong_predictions) >= num_samples // 2:
            break

    visualize_decisions_comparison(correct_predictions, wrong_predictions)

    return correct_predictions, wrong_predictions


def visualize_decisions_comparison(correct_preds, wrong_preds, num_show=5):
    num_correct = min(num_show, len(correct_preds))
    num_wrong = min(num_show, len(wrong_preds))

    fig, axes = plt.subplots(2, num_correct + num_wrong, figsize=(4 * (num_correct + num_wrong), 8))

    for i in range(num_correct):
        sample = correct_preds[i]
        img = sample['image'].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        axes[0, i].imshow(img)
        axes[0, i].set_title(f'True: {sample["true_class"]}\nPred: ✓ {sample["confidence"]:.2f}',
                             color='green', fontsize=9)
        axes[0, i].axis('off')

        if 'heatmap' in sample:
            axes[1, i].imshow(img)
            axes[1, i].imshow(sample['heatmap'], cmap='jet', alpha=0.5)
            axes[1, i].set_title('Attention focus', fontsize=9)
        axes[1, i].axis('off')

    for i in range(num_wrong):
        sample = wrong_preds[i]
        img = sample['image'].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        axes[0, num_correct + i].imshow(img)
        axes[0, num_correct + i].set_title(f'True: {sample["true_class"]}\nPred: {sample["pred_class"]}',
                                           color='red', fontsize=9)
        axes[0, num_correct + i].axis('off')

        if 'heatmap' in sample:
            axes[1, num_correct + i].imshow(img)
            axes[1, num_correct + i].imshow(sample['heatmap'], cmap='jet', alpha=0.5)
            axes[1, num_correct + i].set_title('Attention focus', fontsize=9)
        axes[1, num_correct + i].axis('off')

    plt.suptitle('Сравнение решений: Правильные (зеленые) vs Ошибочные (красные)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/decisions_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("\n" + "=" * 60)
    print("GRAD-CAM АНАЛИЗ И ИНТЕРПРЕТАЦИЯ МОДЕЛИ")
    print("=" * 60)


    print("\nЗагрузка модели...")
    model = create_model(model_type=MODEL_TYPE)
    model.load_state_dict(torch.load(f'./../models/{MODEL_TYPE}/best_model.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    print("\nЗагрузка тестовых данных...")
    test_loader, test_dataset = get_test_loader()

    print("\n1. Анализ точности модели...")
    stats = analyze_model_decisions(model, MODEL_TYPE, test_loader,
                                    test_dataset.classes, DEVICE, num_samples=50)

    print("\n2. Анализ heatmap для различных классов...")
    class_heatmaps, class_attention_stats = analyze_heatmaps_by_class(
        model, MODEL_TYPE, test_loader,
        test_dataset.classes, DEVICE, num_per_class=5
    )

    print("\n3. Интерпретация принятия решений...")
    correct_preds, wrong_preds = interpret_model_decisions(
        model, MODEL_TYPE, test_loader,
        test_dataset.classes, DEVICE, num_samples=20
    )

    print("\n4. Сохранение примеров...")
    save_gradcam_examples(model, MODEL_TYPE, test_loader,
                          test_dataset.classes, DEVICE,
                          save_dir='gradcam_results', num_examples=10)

    print("\n5. Визуализация...")
    for i in range(min(3, len(test_dataset))):
        img_path, label = test_dataset.samples[i]
        image_pil = Image.open(img_path).convert('RGB')
        image_tensor = test_dataset.transform(image_pil).to(DEVICE)

        visualize_single_image(model, MODEL_TYPE, image_tensor, image_pil,
                               test_dataset.classes, save_path=f'results/analysis_{i + 1}.png')

        visualize_top_classes(model, MODEL_TYPE, image_tensor, image_pil,
                              test_dataset.classes, top_k=3, save_path=f'results/top3_{i + 1}.png')

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("Результаты сохранены в папках: results/, gradcam_results/")
    print("Отчеты включают:")
    print("   • heatmaps_by_class.png - визуализация heatmap по классам")
    print("   • decisions_comparison.png - сравнение правильных/неправильных решений")
    print("   • Статистику внимания модели")
    print("   • Интерпретацию принятия решений")
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()