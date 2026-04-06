#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description):
    print(f"\n{'-' * 50}")
    print(f"> {description}")
    print(f"{'-' * 50}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description='Vegetable Classifier Manager')
    parser.add_argument('command', choices=[
        'install', 'install-dev', 'train-resnet50', 'train-custom',
        'convert-onnx', 'convert-all', 'serve-api', 'serve-frontend', 'test',
        'docker-up', 'docker-down', 'clean'
    ], help='Command to execute')

    parser.add_argument('--model-type', type=str, choices=['resnet50', 'custom'],
                        default=None, help='Model type to use (overrides config.py)')

    args = parser.parse_args()

    if args.model_type:
        os.environ['MODEL_TYPE'] = args.model_type

    if args.command == 'install':
        run_command('pip install -r requirements/base.txt', 'Installing base dependencies')

    elif args.command == 'install-dev':
        run_command('pip install -r requirements/dev.txt', 'Installing all dependencies')

    elif args.command == 'train-resnet50':
        run_command('python training/scripts/train.py --model_type resnet50',
                    'Training ResNet50 model')

    elif args.command == 'train-custom':
        run_command('python training/scripts/train.py --model_type custom',
                    'Training custom CNN model')

    elif args.command == 'convert-onnx':
        model_type = args.model_type if args.model_type else 'resnet50'
        run_command(f'python training/scripts/convert_to_onnx.py --model_type {model_type} --force',
                    f'Converting {model_type} model to ONNX')

    elif args.command == 'convert-all':
        run_command('python training/scripts/convert_to_onnx.py --all --force',
                    'Converting all models to ONNX')

    elif args.command == 'serve-api':
        model_type = args.model_type if args.model_type else 'resnet50'
        os.environ['MODEL_TYPE'] = model_type
        print(f"Starting API with MODEL_TYPE={model_type}")
        run_command('uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000',
                    f'Starting API server with {model_type} model')

    elif args.command == 'serve-frontend':
        run_command('uvicorn app.frontend.app:app --reload --host 0.0.0.0 --port 8080',
                    'Starting frontend server')

    elif args.command == 'test':
        run_command('pytest tests/ -v', 'Running tests')

    elif args.command == 'docker-up':
        run_command('docker-compose up --build', 'Starting Docker services')

    elif args.command == 'docker-down':
        run_command('docker-compose down', 'Stopping Docker services')

    elif args.command == 'clean':
        print("Cleaning temporary files...")
        for pattern in ['__pycache__', '*.pyc', '.pytest_cache', 'training/runs']:
            if pattern == 'training/runs':
                import shutil
                if os.path.exists('training/runs'):
                    shutil.rmtree('training/runs')
            else:
                for path in Path('.').rglob(pattern):
                    if path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        print("Clean complete")


if __name__ == '__main__':
    main()