# Data loading and DataLoader pipeline
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
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
