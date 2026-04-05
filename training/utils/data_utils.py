import os
import shutil
import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, USE_AUGMENTATION

def download_dataset(data_dir):
    print("=" * 50)
    print("CHECKING DATASET")
    print("=" * 50)

    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"Dataset already exists at: {data_dir}")
        return data_dir

    print("Downloading dataset from Kaggle...")

    try:
        downloaded_path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")

        source_path = None
        for root, dirs, files in os.walk(downloaded_path):
            if 'train' in dirs and 'validation' in dirs and 'test' in dirs:
                source_path = root
                break

        if source_path is None:
            source_path = downloaded_path

        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

        shutil.copytree(source_path, data_dir)
        print(f"Dataset saved to: {data_dir}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

    return data_dir

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

def create_dataloaders(data_dir):
    if data_dir is None or not os.path.exists(data_dir):
        print("Dataset not found. Downloading...")
        data_dir = download_dataset(data_dir)

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
            raise FileNotFoundError(f"{name} directory not found at {path}")

    train_transform, val_test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names