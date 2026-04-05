"""
Простые функции для тестирования API без лишних деталей
"""
import requests
import mimetypes  # ← добавляем импорт


def ping(api_url="http://localhost:8000"):
    """Проверка статуса сервера"""
    response = requests.get(f"{api_url}/ping")
    return response.json()


def health(api_url="http://localhost:8000"):
    """Проверка статуса сервера"""
    response = requests.get(f"{api_url}/health")
    return response.json()


def classify(api_url, image_path):
    """Классификация изображения"""
    # ✅ Определяем MIME-тип файла
    content_type, _ = mimetypes.guess_type(image_path)
    if content_type is None:
        content_type = 'application/octet-stream'

    with open(image_path, 'rb') as f:
        # ✅ Явно передаём filename и content_type
        response = requests.post(
            f"{api_url}/classify",
            files={'file': (image_path, f, content_type)}
        )
    return response.json()


def similarity(api_url, image1_path, image2_path):
    """Сравнение двух изображений"""
    content_type1, _ = mimetypes.guess_type(image1_path)
    content_type2, _ = mimetypes.guess_type(image2_path)

    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        response = requests.post(
            f"{api_url}/similarity",
            files={
                'file1': (image1_path, f1, content_type1 or 'application/octet-stream'),
                'file2': (image2_path, f2, content_type2 or 'application/octet-stream')
            }
        )
    return response.json()


if __name__ == "__main__":
    API_URL = "http://localhost:8000"

    status = ping(API_URL)
    print(f"Server status: {status}")


    health_info = health(API_URL)
    print(f"Health: {status}")


    result = classify(API_URL, "./dataset/test/Tomato/1001.jpg")
    print(f"Prediction: {result['class_name']} ({result['confidence']:.2%})")

    sim = similarity(API_URL, "./dataset/test/Tomato/1001.jpg", "./dataset/test/Tomato/1001.jpg")
    print(f"Similarity same image: {sim['similarity_percentage']:.1f}%")

    sim = similarity(API_URL, "./dataset/test/Tomato/1019.jpg", "./dataset/test/Pumpkin/1001.jpg")
    print(f"Similarity different image: {sim['similarity_percentage']:.1f}%")