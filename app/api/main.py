from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import IMG_SIZE, DEVICE, MODEL_TYPE, BASE_SAVE_DIR

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. ONNX models disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    model_used: str

class SimilarityResponse(BaseModel):
    similarity_score: float
    similarity_percentage: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    model_type: str
    use_onnx: bool
    device: str
    num_classes: int
    classes: Optional[list] = None

class ModelService:
    def __init__(self, prefer_onnx: bool = True):
        self.prefer_onnx = prefer_onnx and ONNX_AVAILABLE
        self.model = None
        self.onnx_session = None
        self.class_names = None
        self.current_model_path = None
        self.current_model_type = None
        self.use_onnx = False
        self.device = DEVICE

    def find_model(self, model_type: str = MODEL_TYPE) -> Optional[Dict[str, Any]]:
        model_dir = Path(BASE_SAVE_DIR) / model_type

        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return None

        onnx_path = model_dir / "best_model.onnx"
        pth_path = model_dir / "best_model.pth"
        class_names_path = model_dir / "class_names.txt"

        if not class_names_path.exists():
            logger.error(f"Class names not found: {class_names_path}")
            return None

        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]

        if self.prefer_onnx and onnx_path.exists():
            return {
                'type': 'onnx',
                'path': onnx_path,
                'class_names': class_names,
                'model_type': model_type
            }
        elif pth_path.exists():
            return {
                'type': 'pytorch',
                'path': pth_path,
                'class_names': class_names,
                'model_type': model_type
            }

        logger.warning(f"No model found in {model_dir}")
        return None

    def load_onnx_model(self, model_path: Path):
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            providers = [p for p in providers if p in available_providers]

            if not providers:
                providers = ['CPUExecutionProvider']

            self.onnx_session = ort.InferenceSession(str(model_path), providers=providers)
            self.use_onnx = True
            logger.info(f"ONNX model loaded from: {model_path}")
            logger.info(f"Providers: {providers}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False

    def load_pytorch_model(self, model_path: Path, model_type: str):
        try:
            from training.utils.model_utils import create_model

            self.model = create_model(model_type=model_type)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.use_onnx = False
            logger.info(f"PyTorch model loaded from: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False

    def load_model(self, model_type: str = MODEL_TYPE, force_reload: bool = False):
        if not force_reload and (self.model is not None or self.onnx_session is not None):
            logger.info("Model already loaded")
            return True

        model_info = self.find_model(model_type)

        if not model_info:
            logger.error(f"No model found for type: {model_type}")
            return False

        self.class_names = model_info['class_names']
        self.current_model_type = model_info['model_type']
        self.current_model_path = model_info['path']

        if model_info['type'] == 'onnx':
            success = self.load_onnx_model(model_info['path'])
        else:
            success = self.load_pytorch_model(model_info['path'], model_info['model_type'])

        if success:
            logger.info(f"Model info:")
            logger.info(f"Type: {model_info['type'].upper()}")
            logger.info(f"Classes: {len(self.class_names)}")

        return success

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0)

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Извлечение эмбеддинга из модели"""
        if self.onnx_session is not None:
            ort_inputs = {self.onnx_session.get_inputs()[0].name: image_tensor.numpy()}
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            outputs = torch.from_numpy(ort_outputs[0])
            return outputs
        elif self.model is not None:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                return outputs
        else:
            raise RuntimeError("No model loaded")

    def predict(self, image_tensor: torch.Tensor):
        if self.onnx_session is not None:
            ort_inputs = {self.onnx_session.get_inputs()[0].name: image_tensor.numpy()}
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            outputs = torch.from_numpy(ort_outputs[0])
        elif self.model is not None:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
        else:
            raise RuntimeError("No model loaded")

        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

        return top_class.item(), top_prob.item()

    def compute_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Вычисление косинусного сходства между двумя эмбеддингами"""
        with torch.no_grad():
            cos_sim = F.cosine_similarity(tensor1, tensor2, dim=1)
            return cos_sim.item()

    def get_model_info(self) -> dict:
        return {
            'loaded': self.model is not None or self.onnx_session is not None,
            'use_onnx': self.use_onnx,
            'model_path': str(self.current_model_path) if self.current_model_path else None,
            'model_type': self.current_model_type,
            'num_classes': len(self.class_names) if self.class_names else 0
        }

model_service = ModelService(prefer_onnx=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Vegetable Classification API...")
    success = model_service.load_model()
    if success:
        logger.info("API ready for requests")
    else:
        logger.warning("API started without model - please check model files")
    yield
    logger.info("Shutting down API...")

app = FastAPI(
    title="Vegetable Classification API",
    description="API for vegetable classification with automatic ONNX/PyTorch model loading",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/ping", response_model=dict)
async def ping():
    return {
        "status": "alive",
        "model_loaded": model_service.model is not None or model_service.onnx_session is not None,
        "use_onnx": model_service.use_onnx,
        "device": str(model_service.device)
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    model_info = model_service.get_model_info()

    return HealthResponse(
        status="healthy" if model_info['loaded'] else "degraded",
        model_loaded=model_info['loaded'],
        model_path=model_info['model_path'],
        model_type=model_info['model_type'] or MODEL_TYPE,
        use_onnx=model_info['use_onnx'],
        device=str(model_service.device),
        num_classes=model_info['num_classes'],
        classes=model_service.class_names if model_service.class_names else []
    )

@app.post("/classify", response_model=PredictResponse)
async def classify(file: UploadFile = File(...)):
    """Классификация одного изображения"""
    if model_service.model is None and model_service.onnx_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image_tensor = model_service.preprocess_image(image_bytes)

        class_idx, confidence = model_service.predict(image_tensor)
        predicted_class = model_service.class_names[class_idx]

        model_used = "ONNX" if model_service.use_onnx else "PyTorch"

        logger.info(f"Classified as: {predicted_class} (confidence: {confidence:.4f}) using {model_used}")

        return PredictResponse(
            class_name=predicted_class,
            confidence=confidence,
            model_used=model_used
        )

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(
    file1: UploadFile = File(..., description="First image"),
    file2: UploadFile = File(..., description="Second image")
):
    """Вычисление схожести между двумя изображениями"""
    if model_service.model is None and model_service.onnx_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for file in [file1, file2]:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")

    try:
        image1_bytes = await file1.read()
        image2_bytes = await file2.read()

        tensor1 = model_service.preprocess_image(image1_bytes)
        tensor2 = model_service.preprocess_image(image2_bytes)

        embedding1 = model_service.get_embedding(tensor1)
        embedding2 = model_service.get_embedding(tensor2)

        similarity_score = model_service.compute_similarity(embedding1, embedding2)
        similarity_percentage = (similarity_score + 1) / 2 * 100

        logger.info(f"Similarity computed: {similarity_score:.4f} ({similarity_percentage:.2f}%)")

        return SimilarityResponse(
            similarity_score=similarity_score,
            similarity_percentage=similarity_percentage
        )

    except Exception as e:
        logger.error(f"Error during similarity computation: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity error: {str(e)}")

@app.post("/reload")
async def reload_model(model_type: Optional[str] = None):
    """Принудительная перезагрузка модели"""
    try:
        success = model_service.load_model(
            model_type=model_type or MODEL_TYPE,
            force_reload=True
        )

        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_info": model_service.get_model_info()
            }
        else:
            raise HTTPException(status_code=404, detail="No model found to load")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/models")
async def list_models():
    """Список всех доступных моделей"""
    models_info = {}

    for model_type in ['resnet50', 'custom']:
        model_dir = Path(BASE_SAVE_DIR) / model_type

        if model_dir.exists():
            has_onnx = (model_dir / "best_model.onnx").exists()
            has_pth = (model_dir / "best_model.pth").exists()
            has_classes = (model_dir / "class_names.txt").exists()

            models_info[model_type] = {
                'exists': True,
                'has_pytorch': has_pth,
                'has_onnx': has_onnx,
                'has_class_names': has_classes,
                'path': str(model_dir)
            }
        else:
            models_info[model_type] = {'exists': False}

    return models_info

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "service": "Vegetable Classification API",
        "version": "2.0.0",
        "endpoints": {
            "GET /": "This information",
            "GET /ping": "Simple health check",
            "GET /health": "Detailed health information",
            "GET /models": "List all available models",
            "POST /classify": "Classify a single image",
            "POST /similarity": "Compute similarity between two images",
            "POST /reload": "Force reload model"
        },
        "documentation": "/docs",
        "redoc": "/redoc",
        "current_model": model_service.get_model_info() if model_service else None
    }