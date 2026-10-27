"""Shared training/validation loop for Assignment 2."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_bn_eval(model):
    """Keep BatchNorm statistics frozen when the backbone is frozen."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def run_epoch(model, loader, criterion, device, amp, optimizer=None, scaler=None,
              frozen_backbone=False):
    """One pass over ``loader``; trains when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)
    if training and frozen_backbone:
        set_bn_eval(model)

    total_loss = 0.0
    predictions, targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        predictions.append(logits.argmax(1).detach().cpu())
        targets.append(labels.cpu())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "loss": float(total_loss / len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def evaluate(model, loader, criterion, device, amp):
    return run_epoch(model, loader, criterion, device, amp)


def fit(model, loaders, device, epochs, lr=1e-3, weight_decay=1e-4,
        amp=True, checkpoint_path=None, frozen_backbone=False):
    """Train and select the best checkpoint by validation macro-F1."""
    set_seed()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [], "val_macro_f1": [],
        "epoch_time": [],
    }
    best_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train = run_epoch(model, loaders["train"], criterion, device, amp,
                          optimizer=optimizer, scaler=scaler,
                          frozen_backbone=frozen_backbone)
        val = evaluate(model, loaders["val"], criterion, device, amp)
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(train["loss"])
        history["train_accuracy"].append(train["accuracy"])
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["epoch_time"].append(epoch_time)

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss {train['loss']:.4f} acc {train['accuracy']:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
            f"macro-F1 {val['macro_f1']:.4f} | {epoch_time:.1f}s"
        )

    train_time = time.perf_counter() - started
    model.load_state_dict(best_state)
    return history, best_epoch, train_time
