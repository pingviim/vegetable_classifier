from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
from typing import List, Dict
import logging
from contextlib import asynccontextmanager

from config import DEVICE, IMG_SIZE, MODEL_TYPE
from model import create_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = None
class_names = None
device = DEVICE


class PredictResponse(BaseModel):
    class_name: str
    confidence: float

class SimilarityResponse(BaseModel):
    similarity_score: float
    similarity_percentage: float


def load_model_and_classes():
    """Загрузка модели и классов"""
    global model, class_names

    base_models_path = "./models"
    model_folder_path = os.path.join(base_models_path, MODEL_TYPE)

    if not os.path.exists(model_folder_path):
        logger.warning(f"Model folder not found: {model_folder_path}, trying root models folder")
        model_folder_path = base_models_path

    model_path = os.path.join(model_folder_path, "best_model.pth")
    class_names_path = os.path.join(model_folder_path, "class_names.txt")
    config_path = os.path.join(model_folder_path, "config.txt")

    if not os.path.exists(class_names_path):
        logger.error(f"Class names file not found: {class_names_path}")
        return False

    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    model_type = MODEL_TYPE
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'model_type' in line:
                    model_type = line.split(':')[1].strip()
                    break

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Classes: {class_names}")

    try:
        model = create_model(model_type=model_type)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def get_image_tensor(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)
        return input_tensor, image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


def compute_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    with torch.no_grad():
        if hasattr(model, 'gap'):
            x = model.conv1(tensor1)
            x = model.conv2(x)
            x = model.conv3(x)
            x = model.conv4(x)
            x = model.gap(x)
            embedding1 = x.view(x.size(0), -1)

            x = model.conv1(tensor2)
            x = model.conv2(x)
            x = model.conv3(x)
            x = model.conv4(x)
            x = model.gap(x)
            embedding2 = x.view(x.size(0), -1)
        else:
            embedding1 = model.backbone(tensor1)
            embedding2 = model.backbone(tensor2)

            if len(embedding1.shape) > 2:
                embedding1 = F.adaptive_avg_pool2d(embedding1, (1, 1)).view(embedding1.size(0), -1)
                embedding2 = F.adaptive_avg_pool2d(embedding2, (1, 1)).view(embedding2.size(0), -1)

    cos_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
    similarity = cos_sim.item()

    return similarity


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API service...")
    success = load_model_and_classes()
    if not success:
        logger.warning("Model not loaded! Service will run in degraded mode.")
    yield
    logger.info("Shutting down API service...")


# Создаем FastAPI приложение
app = FastAPI(
    title="Vegetable Classification API",
    description="API for vegetable classification and image similarity",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/ping")
async def ping():
    status = {
        "status": "alive",
        "model_loaded": model is not None,
        "device": str(device),
        "num_classes": len(class_names) if class_names else 0
    }

    if model is None:
        status["warning"] = "Model not loaded"

    return JSONResponse(content=status)


@app.post("/classify", response_model=PredictResponse)
async def classify(file: UploadFile = File(...)):
    """
    Классификация одного изображения

    Принимает изображение, возвращает:
    - Предсказанный класс
    - Вероятность предсказания
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    input_tensor, original_image = get_image_tensor(image_bytes)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

        top_prob, top_class = torch.max(probabilities, 1)
        predicted_class = class_names[top_class.item()]
        confidence = top_prob.item()

    logger.info(f"Classified image as: {predicted_class} with confidence {confidence:.4f}")

    return PredictResponse(
        class_name=predicted_class,
        confidence=confidence,

    )

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(
        file1: UploadFile = File(..., description="First image"),
        file2: UploadFile = File(..., description="Second image")
):
    """
    Вычисление схожести между двумя изображениями

    Принимает два изображения, возвращает:
    - Коэффициент схожести (косинусное сходство) от -1 до 1
    - Процент схожести (0-100%)
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for file in [file1, file2]:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")

    image1_bytes = await file1.read()
    image2_bytes = await file2.read()

    tensor1, img1 = get_image_tensor(image1_bytes)
    tensor2, img2 = get_image_tensor(image2_bytes)

    similarity_score = compute_similarity(tensor1, tensor2)
    similarity_percentage = (similarity_score + 1) / 2 * 100

    logger.info(f"Similarity computed: {similarity_score:.4f} ({similarity_percentage:.2f}%)")

    return SimilarityResponse(
        similarity_score=similarity_score,
        similarity_percentage=similarity_percentage
    )

@app.get("/health")
async def health():
    """
    Детальная информация о состоянии сервиса
    """
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "device": str(device),
        "num_classes": len(class_names) if class_names else 0,
        "classes": class_names if class_names else []
    }


@app.get("/")
async def root():
    """
    Корневой эндпоинт с информацией об API
    """
    return {
        "service": "Vegetable Classification API",
        "version": "1.0.0",
        "endpoints": {
            "GET /ping": "Health check",
            "GET /health": "Detailed health information",
            "POST /classify": "Classify a single image",
            "POST /similarity": "Compute similarity between two images"
        },
        "documentation": "/docs",
        "redoc": "/redoc"
    }