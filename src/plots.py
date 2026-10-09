"""Plots saved to the output folder."""

import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC nodes)
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(log_history, path):
    steps = [log["step"] for log in log_history if "loss" in log]
    losses = [log["loss"] for log in log_history if "loss" in log]
    if not losses:
        print("  No training loss was logged (dataset smaller than logging_steps?).")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("Training loss (weighted cross-entropy)")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_epoch_metrics(epochs, scores, best_epoch, selection_set, path):
    """scores: dict metric -> list of values (one per epoch)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric in zip(axes, ("mcc", "accuracy")):
        ax.plot(epochs, scores[metric], marker="o", linewidth=2)
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best epoch: {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper() if metric == "mcc" else metric.capitalize())
        ax.set_title(f"{ax.get_ylabel()} on {selection_set} set")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, path, title="Confusion matrix (test set)"):
    """Row-normalised colours (recall per class) with raw counts written in cells."""
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    n = len(class_names)
    size = max(5, 0.6 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
