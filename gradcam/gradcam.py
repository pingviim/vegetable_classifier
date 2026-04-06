import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        """
        Генерация heatmap

        Args:
            image_tensor: [1, C, H, W] или [C, H, W]
            class_idx: индекс класса (None = предсказанный)

        Returns:
            heatmap: [H, W] numpy array (0-1)
            pred_class: предсказанный класс
            confidence: уверенность
        """

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        was_training = self.model.training
        self.model.train()

        self.model.zero_grad()

        if not image_tensor.requires_grad:
            image_tensor.requires_grad_(True)

        output = self.model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if class_idx is None:
            class_idx = pred_class

        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if not was_training:
            self.model.eval()

        return cam, pred_class, confidence

    def overlay(self, image, heatmap, alpha=0.6):
        """
        Наложение heatmap на изображение
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay


def get_conv_layers(model, model_type):
    """
    Получение сверточных слоев для анализа

    Returns:
        dict: {название_слоя: слой}
    """
    if model_type == 'custom':
        return {
            'conv1': model.conv1,
            'conv2': model.conv2,
            'conv3': model.conv3,
            'conv4': model.conv4
        }
    elif model_type == 'resnet50':
        return {
            'layer2': model.backbone.layer2,
            'layer3': model.backbone.layer3,
            'layer4': model.backbone.layer4
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def visualize_single_image(model, model_type, image_tensor, image_pil, class_names, save_path=None):
    """
    Простая визуализация для одного изображения
    """
    layers = get_conv_layers(model, model_type)
    n_layers = len(layers)

    fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))

    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        prob = F.softmax(output, dim=1)[0, pred].item()

    axes[0, 0].imshow(image_pil)
    axes[0, 0].set_title(f'Original\nPred: {class_names[pred]} ({prob:.2f})', fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for idx, (name, layer) in enumerate(layers.items(), 1):
        gradcam = GradCAM(model, layer)
        heatmap, _, _ = gradcam.generate(image_tensor.unsqueeze(0))
        overlay = gradcam.overlay(image_pil, heatmap)

        axes[0, idx].imshow(overlay)
        axes[0, idx].set_title(f'{name} - overlay', fontsize=10)
        axes[0, idx].axis('off')

        im = axes[1, idx].imshow(heatmap, cmap='jet')
        axes[1, idx].set_title(f'{name} - heatmap', fontsize=10)
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def visualize_top_classes(model, model_type, image_tensor, image_pil, class_names, top_k=3, save_path=None):
    """
    Визуализация топ-k предсказаний модели
    """
    if model_type == 'custom':
        layer = model.conv4
    else:
        layer = model.backbone.layer4

    gradcam = GradCAM(model, layer)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probs = F.softmax(output, dim=1)
        top_probs, top_idx = torch.topk(probs[0], top_k)

    fig, axes = plt.subplots(2, top_k, figsize=(4 * top_k, 8))

    for i in range(top_k):
        class_idx = top_idx[i].item()

        heatmap, _, _ = gradcam.generate(image_tensor.unsqueeze(0), class_idx=class_idx)
        overlay = gradcam.overlay(image_pil, heatmap)

        axes[0, i].imshow(overlay)
        axes[0, i].set_title(f'{class_names[class_idx]}\nprob: {top_probs[i]:.3f}', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(heatmap, cmap='jet')
        axes[1, i].set_title(f'heatmap', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def analyze_model_decisions(model, model_type, dataloader, class_names, device, num_samples=20):
    """
    Простой анализ принятия решений модели

    Returns:
        dict: статистика по классам
    """
    if model_type == 'custom':
        layer = model.conv4
    else:
        layer = model.backbone.layer4

    gradcam = GradCAM(model, layer)
    model.eval()

    stats = {name: {'correct': 0, 'total': 0, 'confidences': []} for name in class_names}

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            labels = labels.to(device)

            for j in range(len(images)):
                true_class = class_names[labels[j].item()]
                stats[true_class]['total'] += 1

                output = model(images[j:j + 1])
                pred = output.argmax(dim=1).item()
                prob = F.softmax(output, dim=1)[0, pred].item()

                if pred == labels[j].item():
                    stats[true_class]['correct'] += 1
                    stats[true_class]['confidences'].append(prob)

    print("\n" + "=" * 50)
    print("Model Decision Analysis:")
    print("=" * 50)

    for class_name in class_names:
        if stats[class_name]['total'] > 0:
            accuracy = stats[class_name]['correct'] / stats[class_name]['total']
            avg_conf = np.mean(stats[class_name]['confidences']) if stats[class_name]['confidences'] else 0
            print(
                f"{class_name:20} | Accuracy: {accuracy:.2f} | Avg Conf: {avg_conf:.3f} | Samples: {stats[class_name]['total']}")

    return stats


def save_gradcam_examples(model, model_type, dataloader, class_names, device, save_dir='gradcam_examples',
                          num_examples=5):
    """
    Сохранение примеров Grad-CAM для отчета
    """
    os.makedirs(save_dir, exist_ok=True)

    if model_type == 'custom':
        layer = model.conv4
    else:
        layer = model.backbone.layer4

    gradcam = GradCAM(model, layer)
    model.eval()

    examples_saved = 0

    for images, labels in dataloader:
        images = images.to(device)

        for j in range(len(images)):
            if examples_saved >= num_examples:
                break

            with torch.no_grad():
                output = model(images[j:j + 1])
                pred = output.argmax(dim=1).item()
                true = labels[j].item()

            heatmap, pred_class, confidence = gradcam.generate(images[j:j + 1])

            img = images[j].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())

            overlay = gradcam.overlay((img * 255).astype(np.uint8), heatmap)

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'True: {class_names[true]}\nPred: {class_names[pred]} ({confidence:.2f})')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(overlay)
            plt.title('Grad-CAM')
            plt.axis('off')

            plt.savefig(f'{save_dir}/example_{examples_saved + 1}.png', dpi=100, bbox_inches='tight')
            plt.close()

            examples_saved += 1

        if examples_saved >= num_examples:
            break

    print(f"Saved {examples_saved} examples to {save_dir}/")


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