"""Classification head, its training loop on cached embeddings, and save/load.

The head has exactly the same layout as Hugging Face's EsmClassificationHead
(dropout -> dense -> tanh -> dropout -> out_proj), and the optimisation
settings mirror the Hugging Face Trainer defaults (AdamW, linear learning-rate
decay without warmup, gradient clipping at 1.0, no weight decay on biases).
Because ESM-2 is frozen, training this head on cached embeddings is the same
method as training EsmForSequenceClassification with a frozen backbone, just
without recomputing ESM-2 for every sequence in every epoch.
"""

import copy
import json
import math
import os

import numpy as np
import torch
from torch import nn

from .embeddings import POOLINGS, get_device

MODEL_CONFIG = "classifier_config.json"
HEAD_WEIGHTS = "head.pt"
FORMAT_VERSION = 1


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.0, initializer_range=0.02):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        for layer in (self.dense, self.out_proj):          # same init as Hugging Face
            nn.init.normal_(layer.weight, mean=0.0, std=initializer_range)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


@torch.no_grad()
def head_logits(head, X, batch_size=4096):
    device = next(head.parameters()).device
    head.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(head(xb).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_head(X_train, y_train, X_sel, y_sel, num_labels, *, epochs, learning_rate,
               batch_size, weight_decay, class_weights, dropout, seed, logging_steps,
               score_fn, initializer_range=0.02, max_grad_norm=1.0):
    """Train the head; evaluate on (X_sel, y_sel) after every epoch.

    Returns a dict with per-epoch head weights, selection-set logits and
    scores, and the training-loss log.
    """
    device = get_device()
    torch.manual_seed(seed)
    head = ClassificationHead(X_train.shape[1], num_labels, dropout, initializer_range).to(device)

    # No weight decay on biases (Hugging Face Trainer convention).
    decay = [p for n, p in head.named_parameters() if not n.endswith("bias")]
    no_decay = [p for n, p in head.named_parameters() if n.endswith("bias")]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
    )
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, (total_steps - step) / total_steps)
    )
    weight = None if class_weights is None else torch.as_tensor(
        class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    epochs_out, loss_log = [], []
    step, running, running_n = 0, 0.0, 0
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(X_t), generator=generator)
        epoch_loss, epoch_n = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx].to(device), y_t[idx].to(device)
            loss = loss_fn(head(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
            running_n += 1
            epoch_loss += loss.item() * len(idx)
            epoch_n += len(idx)
            if step % logging_steps == 0:
                loss_log.append({"step": step, "epoch": epoch, "loss": running / running_n})
                running, running_n = 0.0, 0

        logits = head_logits(head, X_sel)
        epochs_out.append({
            "epoch": epoch,
            "step": step,
            "train_loss": epoch_loss / epoch_n,
            "scores": score_fn(y_sel, logits.argmax(1)),
            "logits": logits,
            "state": copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()}),
        })
    return {"epochs": epochs_out, "loss_log": loss_log, "steps_per_epoch": steps_per_epoch}


def save_model(out_dir, head_state, *, base_model, pooling, max_length, hidden_size,
               class_names, dropout, precision, tokenizer, label_delimiter="|",
               label_field=1, backbone=None):
    """Save a trained classifier. If ``backbone`` is given it is copied into the folder."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(head_state, os.path.join(out_dir, HEAD_WEIGHTS))
    tokenizer.save_pretrained(out_dir)
    if backbone is not None:
        backbone.save_pretrained(os.path.join(out_dir, "backbone"))
    cfg = {
        "format_version": FORMAT_VERSION,
        "base_model": base_model,
        "backbone_included": backbone is not None,
        "pooling": pooling,
        "max_length": max_length,
        "embedding_precision": precision,
        "hidden_size": hidden_size,
        "num_labels": len(class_names),
        "class_names": list(class_names),
        "head_dropout": dropout,
        "label_delimiter": label_delimiter,
        "label_field": label_field,
    }
    with open(os.path.join(out_dir, MODEL_CONFIG), "w") as fh:
        json.dump(cfg, fh, indent=2)


def load_model(model_dir):
    """Return (config dict, head on device, backbone source path/name)."""
    cfg_path = os.path.join(model_dir, MODEL_CONFIG)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found. Pass the 'best_model' folder written by train.py.")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    if cfg["pooling"] not in POOLINGS:
        raise ValueError(f"Unknown pooling '{cfg['pooling']}' in {cfg_path}")
    head = ClassificationHead(cfg["hidden_size"], cfg["num_labels"], cfg["head_dropout"])
    head.load_state_dict(torch.load(os.path.join(model_dir, HEAD_WEIGHTS), map_location="cpu"))
    head.to(get_device()).eval()
    local_backbone = os.path.join(model_dir, "backbone")
    source = local_backbone if os.path.isdir(local_backbone) else cfg["base_model"]
    return cfg, head, source
