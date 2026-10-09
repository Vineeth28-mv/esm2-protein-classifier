"""Classification metrics."""

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


# Order of the metric columns in every output file.
METRIC_KEYS = (
    "mcc",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)

# Short names accepted for selection_metric; they mean the macro average.
METRIC_ALIASES = {"precision": "precision_macro", "recall": "recall_macro", "f1": "f1_macro"}


def compute_all(y_true, y_pred, num_labels):
    """MCC, accuracy, and macro- and weighted-averaged precision/recall/F1.

    Macro: computed per class, then a plain average, so every class counts
    equally and a model that ignores a rare class is penalised.
    Weighted: per-class values averaged in proportion to class size, so it
    reflects performance on the typical sequence.
    MCC is a single number that is robust to class imbalance.
    """
    labels = list(range(num_labels))
    out = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)
    return {k: out[k] for k in METRIC_KEYS}


def format_scores(scores, indent="  "):
    """Readable multi-line text: MCC/accuracy, then macro and weighted rows."""
    lines = [f"{indent}MCC       {scores['mcc']:.4f}    Accuracy {scores['accuracy']:.4f}"]
    for avg in ("macro", "weighted"):
        lines.append(
            f"{indent}{avg:<9s} precision {scores[f'precision_{avg}']:.4f}  "
            f"recall {scores[f'recall_{avg}']:.4f}  F1 {scores[f'f1_{avg}']:.4f}"
        )
    return "\n".join(lines)


def probabilities(logits):
    return softmax(np.asarray(logits, dtype=np.float64), axis=-1)


def report_text(y_true, y_pred, class_names):
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )


def confusion(y_true, y_pred, num_labels):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))


def predictions_frame(seq_ids, logits, class_names, true_labels=None):
    """Per-sequence table: prediction, confidence and a probability per class."""
    import pandas as pd

    probs = probabilities(logits)
    pred_idx = probs.argmax(axis=1)
    class_names = list(class_names)
    df = pd.DataFrame({"seq_id": list(seq_ids)})
    if true_labels is not None:
        df["true_label"] = list(true_labels)
    df["predicted_label"] = [class_names[i] for i in pred_idx]
    if true_labels is not None:
        df["correct"] = df["true_label"] == df["predicted_label"]
    df["confidence"] = probs.max(axis=1)
    for i, name in enumerate(class_names):
        df[f"prob_{name}"] = probs[:, i]
    return df
