import copy
import sys
import time
from pathlib import Path

Root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Root))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from models.Linear import LinearClassifer as Linear
from models.MLP import MLP

from utils.dataloader import load_split, get_dataloaders

DATASET = "fashion_mnist"
SEED = 42
EPOCHS = 10
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess(X_train, X_validate, X_test):
    X_train = X_train.float().div(255.0)
    X_validate = X_validate.float().div(255.0)
    X_test = X_test.float().div(255.0)

    mean = X_train.mean()
    std = X_train.std()
    return ((X_train - mean) / std,(X_validate - mean) / std,(X_test - mean) / std), (mean, std)

def train(model, train_loader, validation_loader, epochs = EPOCHS, lr = LR,device=device, seed = SEED):
    set_seed(seed)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total
        validation_loss, validation_accuracy, _ = evaluate(model, validation_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)


        print(f"Epoch {epoch + 1}/{epochs} "
              f"Train loss: {train_loss:.4f} Train accuracy: {train_accuracy:.4f} "
              f"Validation loss: {validation_loss:.4f} Validation accuracy: {validation_accuracy:.4f}")
    
    return history

@torch.no_grad()
def evaluate(model, dataloader, criterion, device=device):
    model.eval()

    total_loss = 0.0
    predictions, targets = [], []
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)

        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(y.cpu())

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    val_loss = total_loss / len(targets)
    val_accuracy = float((predictions == targets).mean())
    return (val_loss, val_accuracy , float(f1_score(targets, predictions, average="macro")))


