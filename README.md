# Vegetable Classifier

Система классификации овощей на основе глубокого обучения с поддержкой ONNX и REST API.

Результаты экспериментов обучения [Здесь](./ExperimentsResults.md)

## Возможности

- Классификация овощей по фотографии (15 классов)
- Сравнение схожести двух изображений
- Поддержка двух архитектур: ResNet50 и пользовательская CNN
- Автоматическая конвертация моделей в ONNX для быстрого инференса
- REST API с документацией Swagger
- Веб-интерфейс для тестирования
- Docker поддержка для легкого развертывания
- Поддержка GPU (CUDA) и CPU

## Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Обучение моделей](#обучение-моделей)
- [GradCam](#GradCAM-визуализация)
- [API Документация](#api-документация)
- [Веб-интерфейс](#веб-интерфейс)
- [Docker](#docker)
- [Конфигурация](#конфигурация)

## Установка

### Требования

- Python 3.9 или выше
- CUDA 11.8 (опционально, для GPU)
- Docker и Docker Compose (опционально)

### Локальная установка

1. Клонирование репозитория:
```bash
git clone <repository-url>
cd vegetable_classifier
```

2. Создание виртуального окружения:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Установка зависимостей:
Для разработки (все компоненты):
```bash
pip install -r requirements/dev.txt 
```
Только для API:
```bash
pip install -r requirements/api.txt
```
Только для обучения:
```bash
pip install -r requirements/train.txt
```

## Быстрый старт

### Запуск через Python скрипт
```bash
# Установка всех зависимостей
python run.py install-dev

# Обучение модели ResNet50
python run.py train-resnet50

# Конвертация в ONNX
python run.py convert-onnx

# Запуск API сервера
python run.py serve-api

# В другом терминале запуск веб-интерфейса
python run.py serve-frontend
```

### Запуск через Docker
```bash
# Запуск всех сервисов
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d

# Остановка
docker-compose down
```

### Доступ к сервисам
- Веб-интерфейс: http://localhost:8080
- API документация (Swagger): http://localhost:8000/docs
- API документация (ReDoc): http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## Обучение моделей
### Подготовка датасета
Датасет автоматически загружается с Kaggle при первом запуске:
- Источник: [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- Классы: 15 видов овощей
- Разделение: train/validation/test

### Обучение ResNet50
```bash
python training/scripts/train.py --model_type resnet50
```

### Обучение пользовательской CNN
```bash
python training/scripts/train.py --model_type custom
```

### Параметры обучения (config.py)
| Параметр         | Значение по умолчанию | Описание                    |
|------------------|-----------------------|-----------------------------|
| IMG_SIZE         | (224, 224)            | Размер входного изображения |
| NUM_CLASSES      | 15                    | Количество классов          |
| BATCH_SIZE       | 32                    | Размер батча                |
| EPOCHS           | 50                    | 	Количество эпох            |
| LEARNING_RATE    | 0.001                 | Скорость обучения           |
| WEIGHT_DECAY     | 1e-4                  | Регуляризация               |
| DROPOUT_RATE     | 0.5                   | Вероятность dropout         |
| USE_AUGMENTATION | True                  | Использовать аугментацию    |

### Результаты обучения
После обучения в training/models/{model_type}/ сохраняются:
- best_model.pth - веса лучшей модели 
- best_model.onnx - ONNX версия (после конвертации)
- class_names.txt - список классов 
- config.txt - параметры обучения 
- metrics.txt - метрики обучения 
- confusion_matrix.png - матрица ошибок 
- training_history.png - графики обучения 
- sample_predictions.png - примеры предсказаний 
- classification_report.txt - отчет классификации

### Конвертация в ONNX
```bash
# Конвертация последней модели
python training/scripts/convert_to_onnx.py --model_type resnet50

# Конвертация всех моделей
python training/scripts/convert_to_onnx.py --all

# Принудительная переконвертация
python training/scripts/convert_to_onnx.py --model_type resnet50 --force
```

### Инференс изображений
```bash
# Предсказание с ResNet50 моделью
python training/scripts/predict.py --model_type resnet50 --image samples/1.jpg

# Предсказание с custom моделью
python training/scripts/predict.py --model_type custom --image samples/1.jpg

# Интерактивный режим
python training/scripts/predict.py --model_type resnet50
```

## GradCAM визуализация

Модуль GradCAM (Gradient-weighted Class Activation Mapping) для интерпретации решений модели.

### Возможности

- Визуализация областей изображения, которые модель считает важными для классификации
- Анализ внимания модели для разных классов
- Сравнение правильных и ошибочных решений
- Сохранение heatmap для отчета

### Запуск GradCAM анализа

1. Установка зависимостей:
```bash
pip install opencv-python
```
2. Скачивание датасета (если еще не скачан):
```bash
python gradcam/download_dataset_for_test.py
```
3. Запуск анализа:
```bash
python gradcam/test.py
```

### Результаты анализа
После выполнения анализа в папках создаются следующие файлы:
**gradcam_results/**
- heatmaps_by_class_group_*.png - визуализация heatmap по группам классов
- decisions_comparison.png - сравнение правильных и ошибочных решений
- analysis_*.png - анализ отдельных изображений
- top3_*.png - топ-3 предсказания с heatmap

**gradcam_examples/**
- example_*.png - сохраненные примеры Grad-CAM визуализации

### Настройка параметров
В файле gradcam/test.py можно изменить:
```python
MODEL_TYPE = "resnet50"  # или "custom" - тип анализируемой модели
```
### Интерпретация результатов
- Center focus - модель фокусируется на центре объекта (хорошо для цельных объектов)
- Edge focus - модель фокусируется на краях (важно для формы)
- Texture focus - модель использует текстуру для классификации


## API Документация

### Базовый URL
- Локальный запуск: http://localhost:8000
- Docker: http://localhost:8000

### Эндпоинты

**GET /ping**
Простая проверка доступности сервиса.
**Response:**
```json
{
  "status": "alive",
  "model_loaded": true,
  "use_onnx": true,
  "device": "cuda"
}
```

**GET /health**
Детальная информация о состоянии сервиса.
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/training/models/resnet50/best_model.onnx",
  "model_type": "resnet50",
  "use_onnx": true,
  "device": "cpu",
  "num_classes": 15,
  "classes": ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]
}
```

**GET /models**
Список всех доступных моделей.
**Response:**
```json
{
  "resnet50": {
    "exists": true,
    "has_pytorch": true,
    "has_onnx": true,
    "has_class_names": true,
    "path": "/app/training/models/resnet50"
  },
  "custom": {
    "exists": false
  }
}
```

**POST /classify**
Классификация одного изображения.
**Request:**
- file: изображение (multipart/form-data)
- Поддерживаемые форматы: JPEG, PNG, BMP, GIF
**Response:**
```json
{
  "resnet50": {
    "exists": true,
    "has_pytorch": true,
    "has_onnx": true,
    "has_class_names": true,
    "path": "/app/training/models/resnet50"
  },
  "custom": {
    "exists": false
  }
}
```

**POST /classify**
Классификация одного изображения.
**Request:**
- file: изображение (multipart/form-data)
- Поддерживаемые форматы: JPEG, PNG, BMP, GIF
**Response:**
```json
{
  "class_name": "Tomato",
  "confidence": 0.9543,
  "model_used": "ONNX"
}
```
**Пример cURL:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@tomato.jpg"
```

**POST /similarity**
Сравнение двух изображений.
**Request:**
- file1: первое изображение 
- file2: второе изображение
**Response:**
```json
{
  "similarity_score": 0.8765,
  "similarity_percentage": 93.825
}
```
**Пример cURL:**
```bash
curl -X POST "http://localhost:8000/similarity" \
  -F "file1=@image1.jpg" \
  -F "file2=@image2.jpg"
```

**POST /reload**
Принудительная перезагрузка модели (без перезапуска сервера).
**Request body (опционально):**
```json
{
  "model_type": "resnet50"
}
```
**Response:**
```json
{
  "status": "success",
  "message": "Model reloaded successfully",
  "model_info": {
    "loaded": true,
    "use_onnx": true,
    "model_path": "/app/training/models/resnet50/best_model.onnx",
    "model_type": "resnet50",
    "num_classes": 15
  }
}
```

### Примеры использования
```python
import requests

# Классификация
with open('tomato.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': ('tomato.jpg', f, 'image/jpeg')}
    )
result = response.json()
print(f"Predicted: {result['class_name']} ({result['confidence']:.2%})")

# Сравнение
with open('image1.jpg', 'rb') as f1, open('image2.jpg', 'rb') as f2:
    response = requests.post(
        'http://localhost:8000/similarity',
        files={
            'file1': ('image1.jpg', f1, 'image/jpeg'),
            'file2': ('image2.jpg', f2, 'image/jpeg')
        }
    )
similarity = response.json()
print(f"Similarity: {similarity['similarity_percentage']:.1f}%")
```

## Веб-интерфейс
### Режим Classification
1. Выберите режим "Classification"
2. Нажмите на область загрузки или перетащите изображение 
3. Нажмите "Recognize Vegetable"
4. Результат покажет:
   - Название овоща 
   - Уверенность модели в процентах 
   - Использованную модель (ONNX/PyTorch)

### Режим Similarity
1. Выберите режим "Similarity"
2. Загрузите два изображения 
3. Нажмите "Compare Images"
4. Результат покажет процент схожести

### Управление моделями
Веб-интерфейс автоматически использует модель, загруженную в API. Для смены модели используйте API эндпоинт /reload.

## Docker

### Сборка образов
```bash
# Сборка API образа
docker build -f Dockerfile.api -t vegetable-api .

# Сборка фронтенд образа
docker build -f Dockerfile.frontend -t vegetable-frontend .
```
### Запуск через Docker Compose

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./training/models:/app/training/models:ro
    environment:
      - MODEL_TYPE=resnet50
      - USE_ONNX=true
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8080:8080"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
    restart: unless-stopped
```

### Команды Docker
```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f api
docker-compose logs -f frontend

# Остановка
docker-compose down

# Перезапуск конкретного сервиса
docker-compose restart api

# Масштабирование API
docker-compose up -d --scale api=3
```

## Конфигурация

### Файл config.py
Основные параметры конфигурации:
```bash
# Пути
DATA_DIR = "./training/dataset"
BASE_SAVE_DIR = "./training/models"

# Параметры изображений
IMG_SIZE = (224, 224)
NUM_CLASSES = 15

# Параметры обучения
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5
USE_AUGMENTATION = True

# Модель
MODEL_TYPE = 'resnet50'  # или 'custom'

# Устройство
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```