#!/usr/bin/env python
"""Train a protein classifier on frozen ESM-2 embeddings.

Steps
-----
1. Read labelled FASTA files (headers: >sequenceID|label).
2. Compute ESM-2 embeddings once per sequence (or load them from the cache).
3. Train a small classification head on the chosen protein vector
   (pooling: cls = <cls> token, the default; mean = average over residues).
4. Evaluate every epoch, keep the best one, and write results.

Examples
--------
    python train.py --config config.yaml
    python train.py --train my_train.fasta --test my_test.fasta --no-val \
        --pooling mean --epochs 15 --out results/familyX_mean
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoConfig, AutoTokenizer

from src import data as D
from src import metrics as M
from src import plots as P
from src.config import load_config
from src.embeddings import EmbeddingStore
from src.model import ClassificationHead, head_logits, save_model, train_head


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a classifier on frozen ESM-2 embeddings. Options not given on "
                    "the command line are taken from the config file, then from built-in "
                    "defaults.",
    )
    p.add_argument("--config", default="config.yaml",
                   help="YAML config file (default: config.yaml)")
    p.add_argument("--train", help="training FASTA (>id|label)")
    p.add_argument("--test", help="test FASTA (>id|label)")
    p.add_argument("--val", help="optional validation FASTA used to pick the best epoch")
    p.add_argument("--no-val", dest="no_val", action="store_true",
                   help="ignore any validation file in the config")
    p.add_argument("--out", help="output folder")
    p.add_argument("--model", help="ESM-2 model name or local path")
    p.add_argument("--pooling", choices=["cls", "mean"],
                   help="protein vector: cls (<cls> token, default) or mean (average over "
                        "residues)")
    p.add_argument("--max-length", dest="max_length", type=int,
                   help="max tokens per sequence (default: 1024)")
    p.add_argument("--epochs", type=int,
                   help="number of training epochs; if not given, the value in the "
                        "config file is used (10 by default)")
    p.add_argument("--lr", type=float, help="learning rate (default: 1e-3)")
    p.add_argument("--batch-size", dest="batch_size", type=int,
                   help="training batch size (default: 16)")
    p.add_argument("--selection-metric", dest="selection_metric", metavar="METRIC",
                   help="metric used to pick the best epoch: mcc, accuracy, f1_macro, "
                        "precision_macro, recall_macro, f1_weighted, precision_weighted, "
                        "recall_weighted (f1/precision/recall mean macro; default: mcc)")
    p.add_argument("--seed", type=int, help="random seed (default: 42)")
    p.add_argument("--cache-dir", dest="cache_dir",
                   help="folder for cached embeddings (default: embedding_cache)")
    p.add_argument("--no-cache", dest="no_cache", action="store_true",
                   help="do not read or write cached embeddings")
    p.add_argument("--embed-batch-size", dest="embed_batch_size", type=int,
                   help="sequences per batch when computing embeddings (default: 32)")
    p.add_argument("--save-backbone", dest="save_backbone", action="store_true",
                   help="copy ESM-2 into the saved model folder (for offline use)")
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.config):
        config_path = args.config
    elif args.config != "config.yaml":
        sys.exit(f"Config file not found: {args.config}")
    else:
        config_path = None                      # built-in defaults + command line only
    cfg = load_config(config_path, args)

    dcfg, mcfg, ecfg, tcfg, ocfg = (cfg[k] for k in
                                    ("data", "model", "embeddings", "training", "output"))
    out_dir = ocfg["dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    seed = int(tcfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_val = dcfg["val"] is not None
    metric = M.METRIC_ALIASES.get(tcfg["selection_metric"], tcfg["selection_metric"])
    pooling = mcfg["pooling"]
    max_len = int(mcfg["max_length"])
    t0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("  NOTE: no GPU found. Computing embeddings for large datasets will be slow "
              "(but only needs to happen once per dataset).")

    # ------------------------------------------------------------------ data
    header("LOADING DATA")
    rf = dict(delimiter=dcfg["label_delimiter"], label_field=int(dcfg["label_field"]))
    train_ids, train_seqs, train_labels = D.read_fasta(dcfg["train"], **rf)
    test_ids, test_seqs, test_labels = D.read_fasta(dcfg["test"], **rf)
    if use_val:
        val_ids, val_seqs, val_labels = D.read_fasta(dcfg["val"], **rf)

    encoder = LabelEncoder().fit(train_labels)
    class_names = list(encoder.classes_)
    num_labels = len(class_names)
    if num_labels < 2:
        sys.exit(f"Only one class found in the training set ({class_names}). Need at least 2.")
    D.check_labels(class_names, test_labels, "test")
    if use_val:
        D.check_labels(class_names, val_labels, "validation")

    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    y_val = encoder.transform(val_labels) if use_val else None

    header("CLASS DISTRIBUTION")
    print(f"{num_labels} classes: {class_names}")
    D.report_distribution(y_train, class_names, "Training")
    if use_val:
        D.report_distribution(y_val, class_names, "Validation")
    test_counts = D.report_distribution(y_test, class_names, "Test")
    for name, c in zip(class_names, test_counts):
        if c == 0:
            print(f"  WARNING: class '{name}' has no test examples; its test metrics are undefined.")
    print()
    D.report_truncation(train_seqs, max_len, "training")
    D.report_truncation(test_seqs, max_len, "test")
    if use_val:
        D.report_truncation(val_seqs, max_len, "validation")

    # ------------------------------------------------------------ embeddings
    header("ESM-2 EMBEDDINGS")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    store = EmbeddingStore(mcfg["name"], tokenizer, max_len, cache_dir=ecfg["cache_dir"],
                           batch_size=int(ecfg["batch_size"]), fp16=ecfg["fp16"])
    print(f"Model: {mcfg['name']} (frozen) | pooling: {pooling} | max_length: {max_len} | "
          f"precision: {store.precision}")
    print(f"Cache: {ecfg['cache_dir'] or 'disabled'}")
    emb_train, _ = store.get(train_seqs, "training")
    emb_test, _ = store.get(test_seqs, "test")
    emb_val = store.get(val_seqs, "validation")[0] if use_val else None

    X_train, X_test = emb_train[pooling], emb_test[pooling]
    X_val = emb_val[pooling] if use_val else None
    hidden_size = X_train.shape[1]
    print(f"Protein vectors: {hidden_size} numbers each ({pooling} pooling)")

    # --------------------------------------------------------- class weights
    class_weights = None
    if tcfg["class_weighting"] == "balanced":
        header("CLASS WEIGHTS (balanced)")
        class_weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=y_train)
        for name, wi in zip(class_names, class_weights):
            print(f"  {name:<30s} {wi:.4f}")

    # ------------------------------------------------------------- training
    sel_name = "validation" if use_val else "test"
    X_sel, y_sel = (X_val, y_val) if use_val else (X_test, y_test)
    epochs = int(tcfg["epochs"])
    batch_size = int(tcfg["batch_size"])
    steps_per_epoch = -(-len(X_train) // batch_size)
    logging_steps = max(1, min(int(tcfg["logging_steps"]), (steps_per_epoch * epochs) // 20))

    header(f"TRAINING THE CLASSIFICATION HEAD (evaluated on the {sel_name} set every epoch)")
    print(f"Epochs: {epochs} | LR: {tcfg['learning_rate']} | batch size: {batch_size} | "
          f"weight decay: {tcfg['weight_decay']} | selection metric: {metric}")
    # Same dropout as Hugging Face's ESM classification head (0.0 for ESM-2).
    head_dropout = float(getattr(AutoConfig.from_pretrained(mcfg["name"]),
                                 "hidden_dropout_prob", 0.0) or 0.0)

    result = train_head(
        X_train, y_train, X_sel, y_sel, num_labels,
        epochs=epochs, learning_rate=float(tcfg["learning_rate"]), batch_size=batch_size,
        weight_decay=float(tcfg["weight_decay"]), class_weights=class_weights,
        dropout=head_dropout, seed=seed, logging_steps=logging_steps,
        score_fn=lambda yt, yp: M.compute_all(yt, yp, num_labels),
    )
    for e in result["epochs"]:
        s = e["scores"]
        print(f"  Epoch {e['epoch']:>3d}  train loss {e['train_loss']:.4f}  |  {sel_name}: "
              f"MCC={s['mcc']:.4f}  F1(macro)={s['f1_macro']:.4f}  "
              f"F1(weighted)={s['f1_weighted']:.4f}  Acc={s['accuracy']:.4f}")

    rows = [{"epoch": e["epoch"], "step": e["step"], "train_loss": e["train_loss"],
             "evaluated_on": sel_name, **e["scores"]} for e in result["epochs"]]
    results = pd.DataFrame(rows)
    best_idx = int(results[metric].idxmax())                   # ties -> earliest epoch
    best = result["epochs"][best_idx]
    best_epoch, final_epoch = best["epoch"], result["epochs"][-1]["epoch"]
    results["selected"] = results["epoch"] == best_epoch
    results.to_csv(os.path.join(out_dir, "epoch_results.csv"), index=False)

    with open(os.path.join(out_dir, "training_log.json"), "w") as fh:
        json.dump(result["loss_log"], fh, indent=1)
    P.plot_training_loss(result["loss_log"], os.path.join(out_dir, "training_loss.png"))
    P.plot_epoch_metrics(results["epoch"].tolist(),
                         {k: results[k].tolist() for k in ("mcc", "accuracy")},
                         best_epoch, sel_name, os.path.join(out_dir, "epoch_metrics.png"))

    # --------------------------------------------------- final test results
    header("TEST SET RESULTS")
    if use_val:
        head = ClassificationHead(hidden_size, num_labels, head_dropout)
        head.load_state_dict(best["state"])
        head.to(device)
        test_logits = head_logits(head, X_test)
        selection_note = (f"Best epoch ({best_epoch}) selected on the VALIDATION set by {metric}; "
                          f"the test set was evaluated once, with that model only.")
        final_test = None
    else:
        test_logits = best["logits"]
        selection_note = (f"No validation set was provided, so the best epoch ({best_epoch}) was "
                          f"selected on the TEST set by {metric}. Test metrics of the selected "
                          f"epoch may be slightly optimistic; final-epoch ({final_epoch}) test "
                          f"metrics are reported alongside for comparison.")
        final_test = {k: float(results.iloc[-1][k]) for k in M.METRIC_KEYS}

    y_pred = test_logits.argmax(1)
    test_scores = M.compute_all(y_test, y_pred, num_labels)
    print(selection_note)
    print(f"\nBest epoch ({best_epoch}) on the test set:\n" + M.format_scores(test_scores))
    if final_test is not None and final_epoch != best_epoch:
        print(f"\nFinal epoch ({final_epoch}) on the test set, for comparison:\n"
              + M.format_scores(final_test))

    report = M.report_text(y_test, y_pred, class_names)
    print("\n" + report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as fh:
        fh.write(f"Pooling: {pooling} | model: {mcfg['name']}\n{selection_note}\n\n{report}")

    cm = M.confusion(y_test, y_pred, num_labels)
    pd.DataFrame(cm, index=[f"true_{c}" for c in class_names],
                 columns=[f"pred_{c}" for c in class_names]).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv"))
    P.plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"),
                            title=f"Confusion matrix (test set, epoch {best_epoch})")

    preds = M.predictions_frame(test_ids, test_logits, class_names, true_labels=test_labels)
    preds.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

    # ----------------------------------------------------------- save model
    model_dir = os.path.join(out_dir, "best_model")
    save_model(model_dir, best["state"], base_model=mcfg["name"], pooling=pooling,
               max_length=max_len, hidden_size=hidden_size, class_names=class_names,
               dropout=head_dropout, precision=store.precision, tokenizer=tokenizer,
               backbone=store.backbone if ocfg["save_backbone"] else None)

    summary = {
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "selection_set": sel_name,
        "selection_metric": metric,
        "selection_note": selection_note,
        "test_metrics_best_epoch": test_scores,
        "test_metrics_final_epoch": final_test,
        "classes": class_names,
        "pooling": pooling,
        "base_model": mcfg["name"],
        "max_length": max_len,
        "embedding_precision": store.precision,
        "n_train": len(train_seqs),
        "n_val": len(val_seqs) if use_val else 0,
        "n_test": len(test_seqs),
        "runtime_minutes": round((time.time() - t0) / 60, 2),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    header("DONE")
    print(f"Outputs in: {out_dir}/")
    for name, desc in [
        ("summary.json", "headline results and how the best epoch was chosen"),
        ("epoch_results.csv", f"metrics for every epoch on the {sel_name} set"),
        ("epoch_metrics.png", "MCC and accuracy across epochs"),
        ("training_loss.png", "training loss curve"),
        ("classification_report.txt", "per-class precision / recall / F1 on the test set"),
        ("confusion_matrix.csv/.png", "confusion matrix on the test set"),
        ("predictions_test.csv", "per-sequence predictions with class probabilities"),
        ("best_model/", "trained model; use with predict.py"),
    ]:
        print(f"  {name:<28s} {desc}")


if __name__ == "__main__":
    main()
