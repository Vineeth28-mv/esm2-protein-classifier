"""Reading labelled FASTA files and checking labels."""

from collections import Counter

import numpy as np
from Bio import SeqIO


def read_fasta(path, delimiter="|", label_field=1, require_labels=True):
    """Read a FASTA file whose headers look like ``>seqID|label``.

    Returns (ids, sequences, labels). ``labels`` contains None for records
    without a label when ``require_labels`` is False.
    """
    ids, seqs, labels = [], [], []
    missing = []
    n_trailing_stop, internal_stop_ids = 0, []
    print(f"Reading {path} ...")
    for i, record in enumerate(SeqIO.parse(path, "fasta")):
        parts = record.id.split(delimiter)
        seq_id = parts[0]
        label = parts[label_field] if len(parts) > label_field else None
        if label is None or label == "":
            if require_labels:
                missing.append(record.id)
            label = None

        # ESM-2 has no token for '*' (it would become <unk>), so every stop
        # codon is removed, wherever it is: "ANPP*NNNN" -> "ANPPNNNN". Upper-
        # casing is needed because the tokenizer only knows capital letters.
        raw = str(record.seq).upper()
        if "*" in raw:
            if "*" in raw.rstrip("*"):
                internal_stop_ids.append(seq_id)
            else:
                n_trailing_stop += 1
        seq = raw.replace("*", "")

        ids.append(seq_id)
        seqs.append(seq)
        labels.append(label)
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1:,} sequences read ...")

    if not seqs:
        raise ValueError(f"No sequences found in {path}. Is it a FASTA file?")
    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing):,} sequence(s) in {path} have no label in the header "
            f"(expected '>ID{delimiter}label'). Examples: {examples}"
        )

    if n_trailing_stop:
        print(f"  Removed the trailing stop codon '*' from {n_trailing_stop:,} sequence(s).")
    if internal_stop_ids:
        print(f"  NOTE: {len(internal_stop_ids):,} sequence(s) had '*' inside the sequence "
              f"(e.g. {', '.join(internal_stop_ids[:3])}); all '*' were removed. Internal stop "
              f"codons can indicate pseudogenes or frameshifts; you may want to check these.")

    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        print(f"  WARNING: {len(dup):,} duplicated sequence IDs (e.g. {dup[0]}).")
    empty = sum(1 for s in seqs if len(s) == 0)
    if empty:
        raise ValueError(f"{empty:,} empty sequence(s) found in {path}.")

    print(f"  {len(seqs):,} sequences")
    return ids, seqs, labels


def check_labels(train_classes, labels, set_name):
    """Stop with a clear message if a set contains labels unseen in training."""
    unseen = sorted(set(labels) - set(train_classes))
    if unseen:
        raise ValueError(
            f"The {set_name} set contains label(s) that never appear in the training "
            f"set: {unseen}. The model cannot predict a class it was not trained on. "
            f"Add examples of these classes to the training set or remove them."
        )


def report_distribution(labels_encoded, class_names, set_name):
    counts = np.bincount(labels_encoded, minlength=len(class_names))
    total = counts.sum()
    print(f"\n{set_name} set ({total:,} sequences):")
    for name, c in zip(class_names, counts):
        print(f"  {name:<30s} {c:>9,}  ({100 * c / total:5.2f}%)")
    return counts


def report_truncation(seqs, max_length, set_name):
    """ESM-2 adds 2 special tokens, so max_length-2 residues are kept."""
    limit = max_length - 2
    n_long = sum(1 for s in seqs if len(s) > limit)
    if n_long:
        print(
            f"  NOTE: {n_long:,} {set_name} sequence(s) ({100 * n_long / len(seqs):.2f}%) are "
            f"longer than {limit} aa and will be truncated to their first {limit} residues."
        )
    return n_long
