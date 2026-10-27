# Data loading and DataLoader pipeline
from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

# silent the (allegedly) harmless but annoying warnings in assignment 2 
warnings.filterwarnings("ignore", message="Truncated File Read")

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
FOOD101_DIR = Path(DATA_DIR) / "food-101"
FOOD101_MEAN = (0.485, 0.456, 0.406)
FOOD101_STD = (0.229, 0.224, 0.225)
VAL_SIZE = 0.2
SEED = 42
BATCH_SIZE = 64

TV_DATASETS = {
    "fashion_mnist": datasets.FashionMNIST,
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
}


def _to_xy(dataset):
    """Convert a torchvision dataset to the unified numpy format.
        Input:
            - dataset: torchvision dataset object with .data and .targets
        Output:
            - X (np.ndarray): uint8 images of shape (N, C, H, W)
            - y (np.ndarray): int64 labels of shape (N,)
    """
    X = np.asarray(dataset.data)
    y = np.asarray(dataset.targets, dtype=np.int64)
    if X.ndim == 3:
        X = X[:, None, :, :]
    else:
        X = X.transpose(0, 3, 1, 2)
    return X, y


def load_split(dataset_name: str,
               data_dir: str = DATA_DIR,
               val_size: float = VAL_SIZE,
               seed: int = SEED):
    """Load a dataset with torchvision and split the train set into train/validation.
        Input:
            - dataset_name (str): one of 'fashion_mnist', 'mnist', 'cifar10'
            - data_dir (str): directory torchvision downloads the dataset to
            - val_size (float): fraction of the train set used for validation
            - seed (int): random state of the train/validation split
        Output:
            - (X_train, y_train), (X_validate, y_validate), (X_test, y_test): torch tensors,
              X uint8 (N, C, H, W), y int64 (N,)
    """
    if dataset_name not in TV_DATASETS:
        raise ValueError(f"Unknown dataset name {dataset_name!r}; choose one of {sorted(TV_DATASETS)}.")
    train_data = TV_DATASETS[dataset_name](root=data_dir, train=True, download=True)
    test_data = TV_DATASETS[dataset_name](root=data_dir, train=False, download=True)

    X_train_full, y_train_full = _to_xy(train_data)
    X_test, y_test = _to_xy(test_data)

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )
    return (
        (torch.from_numpy(X_train), torch.from_numpy(y_train)),
        (torch.from_numpy(X_validate), torch.from_numpy(y_validate)),
        (torch.from_numpy(np.ascontiguousarray(X_test)), torch.from_numpy(y_test)),
    )


def get_dataloaders(X_train, y_train, X_validate, y_validate, X_test, y_test,
                    batch_size: int = BATCH_SIZE):
    """Build train/validation/test DataLoaders from tensors.
        Input:
            - X_train, y_train, X_validate, y_validate, X_test, y_test (torch.Tensor)
            - batch_size (int): batch size of all loaders
        Output:
            - train_dataloader, validation_dataloader, test_dataloader (DataLoader)
    """
    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        TensorDataset(X_validate, y_validate),
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, validation_dataloader, test_dataloader


# ---------------------------------------------------------------------------
# Food-101 (Assignment 2)
# ---------------------------------------------------------------------------

def download_food101():
    """Download the HF mirror of Food-101 into data/food-101 if missing."""
    parquet = sorted((FOOD101_DIR / "data").glob("*.parquet"))
    if len(parquet) < 11:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="ethz/food101",
            repo_type="dataset",
            local_dir=str(FOOD101_DIR),
            allow_patterns=["data/*.parquet"],
        )
    return FOOD101_DIR


def load_food101():
    """Return the HF dataset: train=75,750, validation=25,250 (official test)."""
    download_food101()
    from datasets import load_dataset

    files = sorted((FOOD101_DIR / "data").glob("*.parquet"))
    return load_dataset(
        "parquet",
        data_files={
            "train": [str(p) for p in files if p.name.startswith("train-")],
            "validation": [str(p) for p in files if p.name.startswith("validation-")],
        },
        cache_dir=str(FOOD101_DIR / "hf-cache"),
    )


class _Food101Subset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.dataset[int(self.indices[i])]
        return self.transform(row["image"].convert("RGB")), int(row["label"])


def make_food101_transform(size, train):
    if train:
        steps = [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps = [
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
        ]
    # [TODO] decide the final augmentation policy for the controlled experiment
    return transforms.Compose(
        steps + [transforms.ToTensor(), transforms.Normalize(FOOD101_MEAN, FOOD101_STD)]
    )


def build_food101_loaders(image_size, batch_size=128, workers=8,
                          val_fraction=0.1, seed=SEED, limit=None):
    """Train/val/test DataLoaders for Food-101.

    The official test split is never re-split.  A stratified ``val_fraction``
    slice of the 75,750 training images is used for validation and model
    selection, matching the Assignment 2 proposal.
    """
    ds = load_food101()
    train_data, test_data = ds["train"], ds["validation"]
    labels = np.asarray(train_data["label"])

    train_idx, val_idx = train_test_split(
        np.arange(len(train_data)),
        test_size=val_fraction,
        stratify=labels,
        random_state=seed,
    )
    test_idx = np.arange(len(test_data))

    # [TODO] remove this smoke-test limiter or replace it with the documented subset rule
    if limit:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(train_idx, min(limit, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, min(limit, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, min(limit, len(test_idx)), replace=False)

    train_set = _Food101Subset(train_data, train_idx, make_food101_transform(image_size, True))
    val_set = _Food101Subset(train_data, val_idx, make_food101_transform(image_size, False))
    test_set = _Food101Subset(test_data, test_idx, make_food101_transform(image_size, False))

    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return {
        "train": DataLoader(train_set, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_set, shuffle=False, **common),
        "test": DataLoader(test_set, shuffle=False, **common),
    }, list(train_data.features["label"].names)
