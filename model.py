import torch.nn as nn
from config import NUM_CLASSES, DROPOUT_RATE, DEVICE, MODEL_TYPE
import torchvision.models as models

class VegetableNN(nn.Module):
    """ Для обучения классификатора была выбрана архитектура состоящая из 4 блоков,
    поскольку большее количество может вызвать переобучение, а меньшее повлияет на точность в худшую сторону.
    Структура блока:
    Свёртка - Обработка - Свёртка - Обработка - Уменьшение картинки - Выключение части нейронов
    Для лучшего выявления текстуры используется 2 свертки. Количество каналов удваивается в каждом блоке (32-64-128-256)
    а Dropout увеличивается (0.1→0.4), чтобы ранние слои учили общие признаки, а поздние не переобучались.
    В конце используется GAP вместо полносвязных слоёв, что уменьшает число параметров и предотвращает запоминание кадров
     """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE * 0.6),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE * 0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(model_type=MODEL_TYPE):
    if model_type == 'custom':
        model = VegetableNN()
        print("Created custom CNN model")
    elif model_type == 'resnet50':
        model = ResNet50(num_classes=NUM_CLASSES)
        print("Created ResNet50 model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)

    return model

