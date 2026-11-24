"""MM-IMDb loading, labels, splits, and datasets for Assignment 3."""

from pathlib import Path
import csv
import json

import numpy as np
import torch
from datasets import load_dataset
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mm-imdb"
CACHE_DIR = DATA_DIR / "hf-cache"
SPLIT_FILE = DATA_DIR / "splits.npz"
GENRES_FILE = DATA_DIR / "genres.json"

IMAGE_SIZE = 224
MAX_TEXT_LEN = 256
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_mmimdb():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        "pranavmr/MM-IMDb",
        split="train",
        cache_dir=str(CACHE_DIR),
    )


def get_genres(dataset):
    if GENRES_FILE.exists():
        return json.loads(GENRES_FILE.read_text())

    genres = sorted({genre for labels in dataset["labels"] for genre in labels})
    GENRES_FILE.write_text(json.dumps(genres, indent=2))
    return genres


def build_label_matrix(dataset, genres):
    labels = np.zeros((len(dataset), len(genres)), dtype=np.float32)
    genre_to_index = {genre: index for index, genre in enumerate(genres)}
    for row_index, row_labels in enumerate(dataset["labels"]):
        for genre in row_labels:
            labels[row_index, genre_to_index[genre]] = 1.0
    return labels


def _deduplicate_splits(splits):
    """Keep one copy per exact record, with priority test > dev > train."""
    duplicate_file = ROOT / "assignment3" / "results" / "exact_duplicate_records.csv"
    if not duplicate_file.exists():
        return splits

    split_of = {}
    for name, indices in splits.items():
        for index in indices:
            split_of[int(index)] = name
    priority = {"train": 0, "dev": 1, "test": 2}

    dropped = set()
    audit_rows = []
    with duplicate_file.open() as handle:
        for row in csv.DictReader(handle):
            first = int(row["id1"])
            second = int(row["id2"])
            if first not in split_of or second not in split_of:
                continue
            first_split = split_of[first]
            second_split = split_of[second]

            if first_split == second_split:
                keep, drop = first, second
            else:
                keep, drop = (
                    (first, second)
                    if priority[first_split] >= priority[second_split]
                    else (second, first)
                )

            dropped.add(drop)
            audit_rows.append(
                {
                    "id1": first,
                    "id2": second,
                    "split_id1": first_split,
                    "split_id2": second_split,
                    "kept": keep,
                    "dropped": drop,
                    "kept_split": split_of[keep],
                    "dropped_split": split_of[drop],
                }
            )

    if dropped:
        audit_file = ROOT / "assignment3" / "results" / "dedup_dropped_indices.csv"
        with audit_file.open("w", newline="") as handle:
            fieldnames = [
                "id1", "id2", "split_id1", "split_id2",
                "kept", "dropped", "kept_split", "dropped_split",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)

    return {
        name: np.array([index for index in indices if int(index) not in dropped])
        for name, indices in splits.items()
    }


def build_splits(dataset, genres, force=False):
    if SPLIT_FILE.exists() and not force:
        data = np.load(SPLIT_FILE)
        splits = {key: data[key] for key in ("train", "dev", "test")}
    else:
        labels = build_label_matrix(dataset, genres)
        x = np.arange(len(dataset))[:, None]

        train_dev_idx, _, test_idx, _ = iterative_train_test_split(
            x, labels, test_size=0.30
        )
        train_idx, _, dev_idx, _ = iterative_train_test_split(
            train_dev_idx, labels[train_dev_idx[:, 0]], test_size=0.10 / 0.70
        )

        splits = {
            "train": train_idx[:, 0],
            "dev": dev_idx[:, 0],
            "test": test_idx[:, 0],
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(SPLIT_FILE, **splits)

    splits = _deduplicate_splits(splits)
    np.savez(SPLIT_FILE, **splits)
    return splits


def image_transform(train, size=IMAGE_SIZE):
    if train:
        steps = [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps = [
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
        ]
    return transforms.Compose(
        steps + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class MMIMDbImageDataset(Dataset):
    def __init__(self, dataset, indices, labels, train):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.transform = image_transform(train)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_index = int(self.indices[index])
        row = self.dataset[row_index]
        image = row["image"].convert("RGB")
        return self.transform(image), torch.from_numpy(self.labels[row_index])


class MMIMDbTextDataset(Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_index = int(self.indices[index])
        row = self.dataset[row_index]
        return row["text"], torch.from_numpy(self.labels[row_index])


class MMIMDbPairDataset(Dataset):
    def __init__(self, dataset, indices, labels, train):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.transform = image_transform(train)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_index = int(self.indices[index])
        row = self.dataset[row_index]
        image = row["image"].convert("RGB")
        return (
            self.transform(image),
            row["text"],
            torch.from_numpy(self.labels[row_index]),
        )


def text_collate(tokenizer):
    def collate(batch):
        texts = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        return encoded, labels

    return collate


def make_image_loader(dataset, indices, labels, batch_size, train, workers=8):
    data = MMIMDbImageDataset(dataset, indices, labels, train=train)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_text_loader(dataset, indices, labels, tokenizer, batch_size, train, workers=8):
    data = MMIMDbTextDataset(dataset, indices, labels)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=text_collate(tokenizer),
    )


def multimodal_collate(tokenizer):
    def collate(batch):
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        labels = torch.stack([item[2] for item in batch])
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        return images, encoded, labels

    return collate


def make_multimodal_loader(dataset, indices, labels, tokenizer, batch_size, train, workers=8):
    data = MMIMDbPairDataset(dataset, indices, labels, train=train)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=multimodal_collate(tokenizer),
    )
