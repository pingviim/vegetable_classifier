.PHONY: help install install-dev train-resnet50 train-custom convert-onnx convert-all serve-api serve-frontend test docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install           - Install base dependencies"
	@echo "  make install-dev       - Install all dependencies (including dev)"
	@echo "  make train-resnet50    - Train ResNet50 model"
	@echo "  make train-custom      - Train custom CNN model"
	@echo "  make convert-onnx      - Convert latest model to ONNX"
	@echo "  make convert-all       - Convert all models to ONNX"
	@echo "  make serve-api         - Run API server locally"
	@echo "  make serve-frontend    - Run frontend server locally"
	@echo "  make test              - Run tests"
	@echo "  make docker-up         - Start all services with Docker"
	@echo "  make docker-down       - Stop all Docker services"
	@echo "  make clean             - Clean temporary files"

install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/dev.txt

train-resnet50:
	python training/scripts/train.py --model_type resnet50

train-custom:
	python training/scripts/train.py --model_type custom

convert-onnx:
	python training/scripts/convert_to_onnx.py --model_type resnet50 --force

convert-all:
	python training/scripts/convert_to_onnx.py --all --force

serve-api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

serve-frontend:
	uvicorn app.frontend.app:app --reload --host 0.0.0.0 --port 8080

test:
	pytest tests/ -v

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf training/runs/