#!/usr/bin/env python
"""Predict classes for new protein sequences with a model trained by train.py.

Examples
--------
    python predict.py --model results/example_run/best_model \
        --fasta new_proteins.fasta --out predictions.csv

Headers can be plain (>seqID) or labelled (>seqID|label). If every sequence
has a label, the metrics are printed as well. The pooling method and
max_length used in training are read from the model folder automatically.
"""

import argparse
import os
import sys

from transformers import AutoTokenizer

from src import data as D
from src import metrics as M
from src.embeddings import EmbeddingStore
from src.model import head_logits, load_model


def parse_args():
    p = argparse.ArgumentParser(description="Predict protein classes with a trained model.")
    p.add_argument("--model", required=True, help="folder saved by train.py (best_model/)")
    p.add_argument("--fasta", required=True, help="input FASTA")
    p.add_argument("--out", default="predictions.csv", help="output CSV (default: predictions.csv)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="sequences per batch when computing embeddings (default: 32)")
    p.add_argument("--cache-dir", default=None,
                   help="optional embedding cache folder (e.g. the one used in training)")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                   help="use half precision on a GPU (--fp16) or not (--no-fp16); by default "
                        "the precision used in training is used")
    p.add_argument("--label-delimiter", default="|")
    p.add_argument("--label-field", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.model):
        sys.exit(f"Model folder not found: {args.model}")

    cfg, head, backbone_source = load_model(args.model)
    class_names = cfg["class_names"]
    print(f"Model: {len(class_names)} classes {class_names}")
    print(f"ESM-2: {cfg['base_model']} | pooling: {cfg['pooling']} | max_length: {cfg['max_length']}"
          + (" | ESM-2 copy included" if cfg.get("backbone_included") else ""))

    ids, seqs, labels = D.read_fasta(args.fasta, delimiter=args.label_delimiter,
                                     label_field=args.label_field, require_labels=False)
    D.report_truncation(seqs, cfg["max_length"], "input")

    # Precision: follow training unless --fp16 / --no-fp16 is given. fp16 is
    # only possible on a GPU; on a CPU full precision (fp32) is always used.
    trained_precision = cfg.get("embedding_precision")
    if args.fp16 is not None:
        fp16 = args.fp16
    elif trained_precision is not None:
        fp16 = trained_precision == "fp16"
    else:
        fp16 = "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The cache key uses the original model name, so a cache made during
    # training is reused even when the ESM-2 copy inside the model folder is used.
    store = EmbeddingStore(cfg["base_model"], tokenizer, cfg["max_length"],
                           cache_dir=args.cache_dir, batch_size=args.batch_size, fp16=fp16)
    print(f"Embedding precision: {store.precision}"
          + (f" (training used {trained_precision})" if trained_precision else ""))
    if trained_precision and store.precision != trained_precision:
        print("  NOTE: precision differs from training (e.g. GPU vs CPU). Predictions can "
              "differ very slightly from those made during training.")
    if backbone_source != cfg["base_model"]:
        from src.embeddings import load_backbone
        store._backbone = load_backbone(backbone_source)
    emb, _ = store.get(seqs, "input")
    logits = head_logits(head, emb[cfg["pooling"]])

    have_labels = all(l is not None for l in labels)
    df = M.predictions_frame(ids, logits, class_names, true_labels=labels if have_labels else None)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nPredictions for {len(df):,} sequences written to {args.out}")
    print("\nPredicted class counts:")
    print(df["predicted_label"].value_counts().to_string())

    if have_labels:
        unknown = sorted(set(labels) - set(class_names))
        if unknown:
            print(f"\nLabels not known to the model, so no metrics are computed: {unknown}")
        else:
            idx = {c: i for i, c in enumerate(class_names)}
            y_true = [idx[l] for l in labels]
            scores = M.compute_all(y_true, logits.argmax(1), len(class_names))
            print("\nMetrics against the labels in the FASTA headers:")
            print(M.format_scores(scores))


if __name__ == "__main__":
    main()
