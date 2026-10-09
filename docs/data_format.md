# Preparing your data

Good data preparation matters more than any model setting. This page covers the file format, building a negative set, and splitting by sequence similarity.

## 1. File format

Plain protein FASTA, with the label after the first `|` in the header:

```text
>WP_012345678.1|familyX
MKKLLIVSAALLAGCSQVNAETIKVGVLHSLSGTMAISETAL
>sp|P0A7V8|RS4_ECOLI|other        <- WRONG: extra '|' characters shift the label
>P0A7V8|other                     <- correct
```

Rules:

- Everything before the first `|` is the ID; the field after it is the label. You can change both with `label_delimiter` and `label_field` in `config.yaml`, or with `--label-delimiter` and `--label-field` on the command line (fields are counted from 0, so `>ID|label` has the label in field 1).
- Anything after the first space in the header is ignored.
- Labels must not contain spaces or `|`. They are case-sensitive (`FamilyX` and `familyX` are different classes).
- Every training class must appear in the training set. A label that appears only in the test or validation set stops the run with a clear message.
- Sequences may be wrapped over several lines. Lower-case letters are upper-cased automatically. Non-standard letters (`X`, `B`, `Z`, `U`, `O`) are allowed. Any other character (e.g. `J`, digits) becomes ESM-2's "unknown" token; the log reports how many sequences contain such characters, and which ones.
- **Stop codons (`*`) are removed wherever they occur**, e.g. `ANPP*NNNN` becomes `ANPPNNNN`, because ESM-2 has no token for `*`. A trailing `*` (as written by Prodigal and many gene callers) is harmless. A `*` *inside* a sequence can mean a pseudogene, a frameshift or a translation error, and removing it joins the pieces on either side, so the log lists such sequences; consider checking or excluding them.

### Adding labels to existing FASTA headers

If you have one FASTA file per class, you can add the label with `sed`:

```bash
# put the label after the ID and drop the rest of the header
sed -E 's/^>([^ ]+).*/>\1|familyX/' familyX.fasta > familyX_labelled.fasta
sed -E 's/^>([^ ]+).*/>\1|other/'   background.fasta > other_labelled.fasta
cat familyX_labelled.fasta other_labelled.fasta > all_labelled.fasta
```

If your headers come from UniProt (`>sp|P0A7V8|RS4_ECOLI ...` or `>tr|A0A0B4J2F2|A0A0B4J2F2_HUMAN ...`), they already contain `|`. Reduce them to the accession first, then add the label, in one command:

```bash
sed -E 's/^>(sp|tr)\|([^|]+)\|[^ ]*/>\2/; s/^>([^ ]+).*/>\1|familyX/' uniprot.fasta > familyX_labelled.fasta
# >sp|P0A7V8|RS4_ECOLI 30S ribosomal ...  becomes  >P0A7V8|familyX
```

Headers that do not start with `sp|` or `tr|` are left as they are by the first part of the command.

**Prediction files need no preparation:** `predict.py` keeps the whole header as the ID unless you pass `--labelled`, so UniProt downloads can be used as they are.

## 2. Building a negative ("other") set

For a "does ESM-2 recognise family X?" experiment, the negatives decide how hard, and how meaningful, the test is.

| Negative set | Difficulty | What a high score tells you |
|---|---|---|
| random proteins from many organisms | easy | ESM-2 separates family X from proteins in general |
| proteins from the same organisms as family X | moderate | the signal is not just organism or composition |
| related families, similar domains or similar length | hard | ESM-2 captures what is specific to family X |

A good set usually mixes all three, with emphasis on the hard ones. Also:

- Try to match the **length distribution** of positives and negatives; otherwise length alone can give the answer away.
- Make sure no true family X members sneak into the negatives (e.g. check with `hmmsearch` against the family's HMM).
- Class imbalance is fine (loss weighting handles it), but keep at least a few hundred positives if you can.

## 3. Split by sequence similarity, not at random

Protein datasets are full of near-duplicates. With a random split, many test proteins have a close relative in the training set, and the model can score well through similarity alone. To measure whether the model **generalises**, cluster first and put whole clusters into each split.

### With MMseqs2

```bash
# 1. Cluster everything at the chosen identity (here 50%) and coverage
mmseqs easy-cluster all_labelled.fasta clusters tmp --min-seq-id 0.5 -c 0.8 --cov-mode 0

# clusters_cluster.tsv now has two columns: representative <TAB> member
```

Then assign **clusters** (not sequences) to train/val/test, for example with this short Python script:

```python
import random
from collections import defaultdict
from Bio import SeqIO

random.seed(42)
clusters = defaultdict(list)
for line in open("clusters_cluster.tsv"):
    rep, member = line.split()
    clusters[rep].append(member)

reps = list(clusters)
random.shuffle(reps)
n = len(reps)
split_of = {}
for i, rep in enumerate(reps):
    split = "train" if i < 0.70 * n else ("val" if i < 0.85 * n else "test")
    for m in clusters[rep]:
        split_of[m] = split

out = {s: open(f"{s}.fasta", "w") for s in ("train", "val", "test")}
for rec in SeqIO.parse("all_labelled.fasta", "fasta"):
    # MMseqs2 uses the header up to the first space as the ID
    out[split_of[rec.id]].write(f">{rec.id}\n{rec.seq}\n")
for fh in out.values():
    fh.close()
```

Afterwards, check that every class appears in every split. A small class can end up entirely in one split by chance; if so, split that class separately (stratify by label), or re-run with another seed.

### Which identity threshold?

| Threshold | Meaning |
|---|---|
| 80–90% | removes near-duplicates only; scores close to "typical use" |
| 50% | moderate; a common choice |
| 30% or lower | strict; tests whether the model finds remote homologs |

Running the same experiment at two thresholds (e.g. 80% and 30%) and comparing MCC shows how much performance depends on similarity to the training data. That is one of the most informative experiments you can do.

## 4. Validation set: optional but recommended

- **With a validation set,** the best epoch is chosen on it and the test set is touched only once. The reported test scores are unbiased.
- **Without one,** the best epoch is chosen on the test set, which is convenient when data is scarce. The outputs state this and also report the final epoch's test scores for comparison.

For small or exploratory datasets, a validation set (10–15% of clusters) is worth the data it costs.
