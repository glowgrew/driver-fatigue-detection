import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .config import EARLY_STOP_PATIENCE, GRAD_CLIP_NORM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PATIENCE, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return improved


def train_epoch(model, loader, optimizer, criterion, device, clip_grad=GRAD_CLIP_NORM):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        n += y.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return loss_sum / n, correct / n, np.concatenate(all_labels), np.concatenate(all_preds)


def train_loop(model, train_loader, val_loader, optimizer, criterion, device,
               num_epochs, early_stopping=None, ckpt_path=None):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        val_f1 = f1_score(y_true, y_pred, average="weighted")
        dt = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"[{epoch:02d}/{num_epochs}] train {tr_loss:.4f}/{tr_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}  F1 {val_f1:.3f}  ({dt:.1f}s)")
        if early_stopping is not None:
            improved = early_stopping(val_f1)
            if improved and ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
            if early_stopping.should_stop:
                print(f"Early stop at epoch {epoch}, best val F1 = {early_stopping.best:.3f}")
                break
        elif ckpt_path is not None:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return history


def compute_metrics(y_true, y_pred, target_names=None, labels=None):
    if labels is None and target_names is not None:
        labels = list(range(len(target_names)))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0,
    )
    p_pc, r_pc, f_pc, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f),
        "f1_per_class": f_pc.tolist(),
        "precision_per_class": p_pc.tolist(),
        "recall_per_class": r_pc.tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            zero_division=0, output_dict=False,
        ),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    # float32 = 4 байта
    n = sum(p.numel() for p in model.parameters())
    return n * 4 / (1024 ** 2)


def measure_fps(fn, n_iters=200, warmup=10):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(n_iters):
        fn()
    elapsed = time.time() - t0
    return n_iters / elapsed if elapsed > 0 else 0.0


def plot_history(history, title, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{title} - loss")
    ax1.legend()
    ax2.plot(history["train_acc"], label="train acc")
    ax2.plot(history["val_acc"], label="val acc")
    if "val_f1" in history:
        ax2.plot(history["val_f1"], label="val F1", linestyle="--")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_title(f"{title} - accuracy / F1")
    ax2.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_confusion(y_true, y_pred, class_names, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def save_metrics_json(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in metrics.items() if k != "report"}
    out["report"] = metrics.get("report", "")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
