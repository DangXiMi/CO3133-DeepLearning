"""Shared helpers for the Assignment 1 experiment notebook.

These are pure helpers: inference timing and figure generation. Training and
evaluation logic stay in ``assignment1/main.ipynb``.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def measure_inference_time(model, dataloader, device, warmup_batches=2):
    """Measure a full forward pass over ``dataloader`` after a short warmup.

    Returns:
        elapsed seconds, milliseconds per sample.
    """
    model.eval()
    iterator = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            X, _ = next(iterator)
        except StopIteration:
            break
        model(X.to(device))

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    total = 0
    for X, _ in dataloader:
        model(X.to(device))
        total += X.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed, 1000.0 * elapsed / total if total else float("nan")


def plot_curves(history, path, title):
    """Plot train/validation loss, accuracy, and validation macro-F1."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", markersize=3, label="train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", markersize=3, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", markersize=3, label="train")
    axes[1].plot(epochs, history["validation_accuracy"], marker="o", markersize=3, label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["validation_macro_f1"], marker="o", markersize=3, label="validation")
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("macro-F1")
    axes[2].legend()

    best_epoch = history.get("best_epoch", 0)
    if best_epoch:
        for ax in axes:
            ax.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1)

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle(f"{title} — best epoch {best_epoch} by {history.get('monitor')}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, class_names, path, title):
    """Plot a labelled confusion matrix."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_examples(X_test, targets, predictions, class_names, path, title, mean, std):
    """Plot five correct and five incorrect test predictions."""
    correct = np.where(predictions == targets)[0]
    incorrect = np.where(predictions != targets)[0]
    chosen = list(correct[:5]) + list(incorrect[:5])

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    for ax in axes.ravel():
        ax.axis("off")

    for position, index in enumerate(chosen[:10]):
        ax = axes.ravel()[position]
        image = X_test[index].cpu().numpy()
        if image.shape[0] == 1:
            image = image[0]
        else:
            image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * std + mean, 0.0, 1.0)

        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        is_correct = predictions[index] == targets[index]
        ax.set_title(
            f"T: {class_names[targets[index]]}\nP: {class_names[predictions[index]]}",
            color="green" if is_correct else "red",
            fontsize=8,
        )

    fig.suptitle(f"{title} — top row: correct, bottom row: incorrect")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
