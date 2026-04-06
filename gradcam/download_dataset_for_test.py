import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kagglehub
import shutil

DATA_DIR = "./training/dataset"

def download_dataset():
    print("=" * 50)
    print("CHECKING DATASET")
    print("=" * 50)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, DATA_DIR)

    if os.path.exists(dataset_path):
        if len(os.listdir(dataset_path)) > 0:
            print(f"Dataset already exists at: {dataset_path}")
            return dataset_path
        else:
            print(f"Dataset directory exists but is empty. Re-downloading...")
            os.rmdir(dataset_path)

    print("Downloading dataset from Kaggle...")

    try:
        downloaded_path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")
        print(f"Dataset downloaded to: {downloaded_path}")

        print(f"Checking downloaded dataset structure...")
        source_path = None
        for root, dirs, files in os.walk(downloaded_path):
            if 'train' in dirs and 'validation' in dirs and 'test' in dirs:
                source_path = root
                break

        if source_path is None:
            source_path = downloaded_path
            print(f"Using root directory as source: {source_path}")
        else:
            print(f"Found dataset structure in: {source_path}")

        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

        print(f"Copying dataset to: {dataset_path}")
        shutil.copytree(source_path, dataset_path)

        print(f"Dataset successfully saved to: {dataset_path}")

        train_path = os.path.join(dataset_path, 'train')
        val_path = os.path.join(dataset_path, 'validation')
        test_path = os.path.join(dataset_path, 'test')

        if os.path.exists(train_path):
            num_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
            print(f"Dataset info:")
            print(f"   - Classes: {num_classes}")
            print(f"   - Train samples: {sum(len(files) for _, _, files in os.walk(train_path))}")
            print(
                f"   - Validation samples: {sum(len(files) for _, _, files in os.walk(val_path)) if os.path.exists(val_path) else 'N/A'}")
            print(
                f"   - Test samples: {sum(len(files) for _, _, files in os.walk(test_path)) if os.path.exists(test_path) else 'N/A'}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed: pip install kagglehub")
        print("2. Check your internet connection")
        print("3. Try manually downloading from: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset")
        print(f"4. Manually place the dataset in: {dataset_path}")
        print("The folder should contain 'train', 'validation', and 'test' subfolders")
        raise

    return dataset_path


if __name__ == "__main__":
    download_dataset()