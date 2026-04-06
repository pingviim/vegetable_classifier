import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import onnx
import onnxruntime as ort
import argparse
from pathlib import Path

from config import BASE_SAVE_DIR, IMG_SIZE


def find_all_models(models_base_dir=None):
    """Поиск всех моделей .pth в структуре папок training/models/{model_type}/best_model.pth"""
    if models_base_dir is None:
        models_base_dir = Path(BASE_SAVE_DIR)
    else:
        models_base_dir = Path(models_base_dir)

    models_found = []

    # Проверяем оба типа моделей: resnet50 и custom
    for model_type in ['resnet50', 'custom']:
        model_dir = models_base_dir / model_type

        if not model_dir.exists():
            continue

        pth_path = model_dir / "best_model.pth"
        if not pth_path.exists():
            continue

        models_found.append({
            'pth_path': pth_path,
            'onnx_path': model_dir / "best_model.onnx",
            'model_type': model_type,
            'model_dir': model_dir
        })

    return models_found


def convert_pytorch_to_onnx(pth_path, onnx_path, model_type='resnet50', input_size=(3, 224, 224), force=False):
    """Конвертация PyTorch модели в ONNX формат"""

    if onnx_path.exists() and not force:
        print(f"⏭️  ONNX already exists: {onnx_path}")
        return onnx_path

    from training.utils.model_utils import create_model

    print(f"🔄 Converting: {pth_path}")
    print(f"   Model type: {model_type}")

    try:

        device = torch.device('cpu')
        print(f"   Loading model on CPU...")
        model = create_model(model_type=model_type)
        checkpoint = torch.load(pth_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        dummy_input = torch.randn(1, *input_size).to(device)

        print(f"   Exporting to ONNX...")

        # Конвертируем в ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )

        # Проверяем ONNX модель
        print(f"   Validating ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Тестируем инференс на CPU
        print(f"   Testing inference...")
        ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        print(f"   ✅ Success! Output shape: {ort_outputs[0].shape}")
        return onnx_path

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_all_models(force=False):
    """Конвертация всех найденных моделей"""
    models = find_all_models()

    if not models:
        print("❌ No models found in training/models/resnet50/ or training/models/custom/")
        print("   Expected structure: training/models/{resnet50|custom}/best_model.pth")
        return

    print(f"\n📦 Found {len(models)} model(s) to convert\n")

    converted = 0
    failed = 0

    for i, model_info in enumerate(models, 1):
        print(f"\n{'=' * 50}")
        print(f"[{i}/{len(models)}] Processing {model_info['model_type']}")
        print(f"{'=' * 50}")
        result = convert_pytorch_to_onnx(
            model_info['pth_path'],
            model_info['onnx_path'],
            model_type=model_info['model_type'],
            force=force
        )
        if result:
            converted += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"✅ Converted: {converted}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(models)} models")
    print(f"{'=' * 50}")


def convert_model_type(model_type='resnet50', force=False):
    """Конвертация модели указанного типа"""
    model_dir = Path(BASE_SAVE_DIR) / model_type

    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print(f"   Expected structure: training/models/{model_type}/")
        return None

    pth_path = model_dir / "best_model.pth"
    if not pth_path.exists():
        print(f"❌ best_model.pth not found in {model_dir}")
        return None

    onnx_path = model_dir / "best_model.onnx"

    return convert_pytorch_to_onnx(pth_path, onnx_path, model_type, force=force)


def convert_specific_model(model_path, model_type=None, force=False):
    """Конвертация конкретной модели по пути"""
    pth_path = Path(model_path)

    if not pth_path.exists():
        print(f"❌ Model not found: {pth_path}")
        return None

    # Автоматически определяем тип модели из пути
    if model_type is None:
        if 'resnet50' in str(pth_path).lower():
            model_type = 'resnet50'
        elif 'custom' in str(pth_path).lower():
            model_type = 'custom'
        else:
            model_type = 'resnet50'  # по умолчанию

    onnx_path = pth_path.parent / "best_model.onnx"

    return convert_pytorch_to_onnx(pth_path, onnx_path, model_type, force=force)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert ResNet50 model
  python convert_to_onnx.py --model_type resnet50

  # Convert custom model
  python convert_to_onnx.py --model_type custom

  # Convert both models
  python convert_to_onnx.py --all

  # Convert specific model
  python convert_to_onnx.py --model_path training/models/resnet50/best_model.pth

  # Force reconvert (overwrite existing ONNX)
  python convert_to_onnx.py --model_type resnet50 --force
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Convert all models from both resnet50 and custom folders')
    parser.add_argument('--model_type', type=str, choices=['resnet50', 'custom'],
                        default='resnet50',
                        help='Model type to convert (default: resnet50)')
    parser.add_argument('--force', action='store_true',
                        help='Force reconversion even if ONNX file exists')
    parser.add_argument('--model_path', type=str,
                        help='Path to specific .pth file to convert')

    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print("🎯 PyTorch to ONNX Converter")
    print(f"{'=' * 50}")
    print(f"Base models directory: {BASE_SAVE_DIR}")

    if args.model_path:
        # Конвертация конкретной модели
        print(f"Converting specific model: {args.model_path}")
        convert_specific_model(args.model_path, force=args.force)

    elif args.all:
        # Конвертация всех моделей
        print("Converting all models...")
        convert_all_models(force=args.force)

    else:
        # Конвертация модели указанного типа
        print(f"Converting {args.model_type} model...")
        convert_model_type(model_type=args.model_type, force=args.force)

    print(f"\n{'=' * 50}")
    print("✅ Done!")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()