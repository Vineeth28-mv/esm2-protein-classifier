"""Reading FASTA files and checking labels."""

from collections import Counter

import numpy as np
from Bio import SeqIO


def read_fasta(path, delimiter="|", label_field=1, require_labels=True, split_labels=True):
    """Read a FASTA file.

    split_labels=True (training, or prediction with --labelled): the header is
    split at ``delimiter``; field ``label_field`` (counted from 0) is the label
    and the remaining field(s) form the ID. So both ``>seqID|label`` (default,
    label_field=1) and ``>label;seqID`` (delimiter ";", label_field=0) give the
    ID "seqID".

    split_labels=False (prediction without labels): the whole header word
    (everything before the first space) is the ID, so headers such as
    ``>sp|P0A7V8|RS4_ECOLI`` stay intact; all labels are None.

    Returns (ids, sequences, labels).
    """
    ids, seqs, labels = [], [], []
    missing = []
    n_trailing_stop, internal_stop_ids, only_stop_ids, empty_ids = 0, [], [], []
    n_extra_delims = 0
    print(f"Reading {path} ...")
    for i, record in enumerate(SeqIO.parse(path, "fasta")):
        if split_labels:
            parts = record.id.split(delimiter)
            # A normal header has exactly two fields: an ID and a label.
            if len(parts) > 2:
                n_extra_delims += 1
            label = parts[label_field] if len(parts) > label_field else None
            # The ID is everything except the label field.
            id_parts = [p for j, p in enumerate(parts) if j != label_field]
            seq_id = delimiter.join(id_parts) if id_parts else record.id
            if label is None or label == "":
                if require_labels:
                    missing.append(record.id)
                label = None
        else:
            seq_id, label = record.id, None

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
        if not seq:
            (only_stop_ids if raw else empty_ids).append(seq_id)

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
    if only_stop_ids:
        raise ValueError(
            f"{len(only_stop_ids):,} sequence(s) in {path} contain only stop codons ('*'), so "
            f"nothing is left after removing them: {', '.join(only_stop_ids[:5])}"
        )
    if empty_ids:
        raise ValueError(f"{len(empty_ids):,} empty sequence(s) found in {path}: "
                         f"{', '.join(empty_ids[:5])}")

    if n_trailing_stop:
        print(f"  Removed the trailing stop codon '*' from {n_trailing_stop:,} sequence(s).")
    if internal_stop_ids:
        print(f"  NOTE: {len(internal_stop_ids):,} sequence(s) had '*' inside the sequence "
              f"(e.g. {', '.join(internal_stop_ids[:3])}); all '*' were removed. Internal stop "
              f"codons can indicate pseudogenes or frameshifts; you may want to check these.")

    if n_extra_delims:
        print(f"  WARNING: {n_extra_delims:,} header(s) contain more than one '{delimiter}' "
              f"(e.g. UniProt '>sp|P0A7V8|RS4_ECOLI'), so the label may have been read from the "
              f"wrong part of the header; see docs/data_format.md.")
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        print(f"  WARNING: {len(dup):,} sequence ID(s) occur more than once (e.g. '{dup[0]}'), "
              f"so rows in the output cannot all be told apart.")

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
