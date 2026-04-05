import os
import shutil
import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, USE_AUGMENTATION, DATA_DIR

### Загрузка датасета
def download_dataset():
    print("=" * 50)
    print("CHECKING DATASET")
    print("=" * 50)

    project_root = os.path.dirname(os.path.abspath(__file__))
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
            print(f"📊Dataset info:")
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

### Нормализация и Аугментация данных
def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if USE_AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            normalize
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, val_test_transform

### Загрузчики данных
def create_dataloaders(data_dir=None):
    if data_dir is None or not os.path.exists(data_dir):
        print("Dataset not found. Downloading...")
        data_dir = download_dataset()

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')
    test_path = os.path.join(data_dir, 'test')

    if not os.path.exists(train_path):
        for root, dirs, files in os.walk(data_dir):
            if 'train' in dirs and 'validation' in dirs and 'test' in dirs:
                data_dir = root
                train_path = os.path.join(data_dir, 'train')
                val_path = os.path.join(data_dir, 'validation')
                test_path = os.path.join(data_dir, 'test')
                break

    for path, name in [(train_path, 'train'), (val_path, 'validation'), (test_path, 'test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} directory not found at {path}\n"
                f"Dataset should contain 'train', 'validation', and 'test' folders"
            )

    print(f"Loading dataset from: {data_dir}")

    train_transform, val_test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names