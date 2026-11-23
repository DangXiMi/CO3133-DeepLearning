"""Training and evaluation for Assignment 3 M2."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tune_thresholds(probabilities, targets, grid=np.arange(0.05, 0.96, 0.05)):
    thresholds = np.full(probabilities.shape[1], 0.5, dtype=np.float32)
    for index in range(probabilities.shape[1]):
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in grid:
            predictions = probabilities[:, index] >= threshold
            score = f1_score(targets[:, index], predictions, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[index] = best_threshold
    return thresholds


def metrics_from_probabilities(probabilities, targets, thresholds):
    predictions = (probabilities >= thresholds[None, :]).astype(np.int32)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
        "exact_match": float(accuracy_score(targets, predictions)),
        "per_class_f1": [float(value) for value in per_class_f1],
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in recall],
        "per_class_support": [int(value) for value in support],
    }


def run_image_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    probabilities, targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        targets.append(labels.detach().cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    return float(total_loss / len(targets)), probabilities, targets


def run_text_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    probabilities, targets = [], []

    for encoded, labels in loader:
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        targets.append(labels.detach().cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    return float(total_loss / len(targets)), probabilities, targets


def fit_image(model, loaders, device, epochs, lr=3e-4, weight_decay=1e-4,
              amp=True, seed=42, checkpoint_path=None):
    set_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "epoch_time": []}
    best_f1 = -1.0
    best_epoch = 0
    best_thresholds = np.full(23, 0.5, dtype=np.float32)
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, _, _ = run_image_epoch(
            model, loaders["train"], criterion, device, amp,
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_probabilities, val_targets = run_image_epoch(
            model, loaders["dev"], criterion, device, amp
        )
        thresholds = tune_thresholds(val_probabilities, val_targets)
        val_metrics = metrics_from_probabilities(val_probabilities, val_targets, thresholds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_time"].append(time.perf_counter() - start)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = thresholds
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)

        print(
            f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"{history['epoch_time'][-1]:.1f}s"
        )

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history, best_epoch, best_thresholds


def fit_text(model, loaders, device, epochs, lr=2e-5, weight_decay=1e-4,
             amp=True, seed=42, checkpoint_path=None):
    set_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "epoch_time": []}
    best_f1 = -1.0
    best_epoch = 0
    best_thresholds = np.full(23, 0.5, dtype=np.float32)
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, _, _ = run_text_epoch(
            model, loaders["train"], criterion, device, amp,
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_probabilities, val_targets = run_text_epoch(
            model, loaders["dev"], criterion, device, amp
        )
        thresholds = tune_thresholds(val_probabilities, val_targets)
        val_metrics = metrics_from_probabilities(val_probabilities, val_targets, thresholds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_time"].append(time.perf_counter() - start)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = thresholds
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)

        print(
            f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"{history['epoch_time'][-1]:.1f}s"
        )

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history, best_epoch, best_thresholds


@torch.no_grad()
def extract_image_features(encoder, loader, device, amp=True):
    encoder.eval()
    features, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            output = encoder(images)
        features.append(output.float().cpu())
        targets.append(labels)
    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def extract_text_features(encoder, loader, device, amp=True):
    encoder.eval()
    features, targets = [], []
    for encoded, labels in loader:
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        features.append(output.last_hidden_state[:, 0, :].float().cpu())
        targets.append(labels)
    return torch.cat(features), torch.cat(targets)


def run_feature_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    probabilities, targets = [], []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=scaler is not None and scaler.is_enabled()):
                logits = model(features)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        targets.append(labels.detach().cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    return float(total_loss / len(targets)), probabilities, targets


def fit_fusion(head, train_features, train_labels, dev_features, dev_labels,
               device, epochs=30, lr=1e-3, weight_decay=1e-4, batch_size=256,
               seed=42, checkpoint_path=None):
    set_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        TensorDataset(dev_features, dev_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "epoch_time": []}
    best_f1 = -1.0
    best_epoch = 0
    best_thresholds = np.full(23, 0.5, dtype=np.float32)
    best_state = copy.deepcopy(head.state_dict())

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, _, _ = run_feature_epoch(
            head, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_probabilities, val_targets = run_feature_epoch(
            head, dev_loader, criterion, device
        )
        thresholds = tune_thresholds(val_probabilities, val_targets)
        val_metrics = metrics_from_probabilities(val_probabilities, val_targets, thresholds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_time"].append(time.perf_counter() - start)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = thresholds
            best_state = copy.deepcopy(head.state_dict())
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | "
                f"val macro-F1 {val_metrics['macro_f1']:.4f}"
            )

    head.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history, best_epoch, best_thresholds


def run_gmu_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None,
                  drop_mode=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    probabilities, targets = [], []
    for image_features, text_features, labels in loader:
        image_features = image_features.to(device, non_blocking=True)
        text_features = text_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        if drop_mode == "image":
            drop_image = torch.ones(batch_size, device=device)
            drop_text = torch.zeros(batch_size, device=device)
        elif drop_mode == "text":
            drop_image = torch.zeros(batch_size, device=device)
            drop_text = torch.ones(batch_size, device=device)
        else:
            drop_image = drop_text = None

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(
                    image_features,
                    text_features,
                    drop_image=drop_image,
                    drop_text=drop_text,
                )
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        targets.append(labels.detach().cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    return float(total_loss / len(targets)), probabilities, targets


def fit_gmu(model, train_image_features, train_text_features, train_labels,
            dev_image_features, dev_text_features, dev_labels, device,
            epochs=30, lr=1e-3, weight_decay=1e-4, batch_size=256,
            modality_dropout=0.0, seed=42, checkpoint_path=None):
    set_seed(seed)
    model.modality_dropout = modality_dropout

    train_loader = DataLoader(
        TensorDataset(train_image_features, train_text_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        TensorDataset(dev_image_features, dev_text_features, dev_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "epoch_time": []}
    best_f1 = -1.0
    best_epoch = 0
    best_thresholds = np.full(23, 0.5, dtype=np.float32)
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, _, _ = run_gmu_epoch(
            model, train_loader, criterion, device, torch.cuda.is_available(),
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_probabilities, val_targets = run_gmu_epoch(
            model, dev_loader, criterion, device, torch.cuda.is_available()
        )
        thresholds = tune_thresholds(val_probabilities, val_targets)
        val_metrics = metrics_from_probabilities(val_probabilities, val_targets, thresholds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_time"].append(time.perf_counter() - start)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = thresholds
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | "
                f"val macro-F1 {val_metrics['macro_f1']:.4f}"
            )

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history, best_epoch, best_thresholds


def run_multimodal_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None,
                         drop_mode=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    probabilities, targets = [], []
    for images, encoded, labels in loader:
        images = images.to(device, non_blocking=True)
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        if drop_mode == "image":
            drop_image = torch.ones(batch_size, device=device)
            drop_text = torch.zeros(batch_size, device=device)
        elif drop_mode == "text":
            drop_image = torch.zeros(batch_size, device=device)
            drop_text = torch.ones(batch_size, device=device)
        else:
            drop_image = drop_text = None

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(
                    images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drop_image=drop_image,
                    drop_text=drop_text,
                )
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        probabilities.append(torch.sigmoid(logits).detach().cpu())
        targets.append(labels.detach().cpu())

    probabilities = torch.cat(probabilities).numpy()
    targets = torch.cat(targets).numpy()
    return float(total_loss / len(targets)), probabilities, targets


def fit_multimodal(model, loaders, device, epochs=3, image_lr=1e-5, text_lr=1e-5,
                   fusion_lr=1e-4, weight_decay=1e-4, amp=True, seed=42,
                   checkpoint_path=None):
    set_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": image_lr},
            {"params": model.text_encoder.parameters(), "lr": text_lr},
            {"params": model.gmu.parameters(), "lr": fusion_lr},
        ],
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "epoch_time": []}
    best_f1 = -1.0
    best_epoch = 0
    best_thresholds = np.full(23, 0.5, dtype=np.float32)
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, _, _ = run_multimodal_epoch(
            model, loaders["train"], criterion, device, amp,
            optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_probabilities, val_targets = run_multimodal_epoch(
            model, loaders["dev"], criterion, device, amp
        )
        thresholds = tune_thresholds(val_probabilities, val_targets)
        val_metrics = metrics_from_probabilities(val_probabilities, val_targets, thresholds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_time"].append(time.perf_counter() - start)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = thresholds
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)

        print(
            f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"{history['epoch_time'][-1]:.1f}s"
        )

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history, best_epoch, best_thresholds
