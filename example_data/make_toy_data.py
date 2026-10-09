#!/usr/bin/env python
"""Generate the small SYNTHETIC demo dataset in this folder.

These sequences are NOT real proteins. They exist only so you can check that
the pipeline runs end to end in a few minutes. Do not draw biological
conclusions from results on this data.

Design (mimics a "does ESM-2 recognise family X?" experiment):
  * "familyX": variants of one random ancestor sequence (~35% of positions
    mutated, plus small insertions/deletions).
  * "other":   variants of five unrelated random ancestors.
Each ancestor is split so that train / val / test variants are generated
independently, but they still share an ancestor (as real family members do).

Run from the repository root:
    python example_data/make_toy_data.py
"""

import os
import random

AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 7
OUT = os.path.dirname(os.path.abspath(__file__))

# sequences per label in each split
COUNTS = {"train": 120, "val": 30, "test": 40}


def random_protein(rng, length):
    return "M" + "".join(rng.choice(AA) for _ in range(length - 1))


def mutate(rng, seq, sub_rate=0.35, indel_rate=0.03):
    out = []
    for i, aa in enumerate(seq):
        if i > 0 and rng.random() < indel_rate:   # deletion
            continue
        out.append(rng.choice(AA) if (i > 0 and rng.random() < sub_rate) else aa)
        if rng.random() < indel_rate:              # insertion
            out.append(rng.choice(AA))
    return "".join(out)


def main():
    rng = random.Random(SEED)
    family_x = random_protein(rng, rng.randint(180, 260))
    others = [random_protein(rng, rng.randint(120, 350)) for _ in range(5)]

    n = 0
    for split, count in COUNTS.items():
        records = []
        for _ in range(count):
            n += 1
            records.append((f"toyX_{n:04d}", "familyX", mutate(rng, family_x)))
        for _ in range(count):
            n += 1
            records.append((f"toyO_{n:04d}", "other", mutate(rng, rng.choice(others))))
        rng.shuffle(records)

        path = os.path.join(OUT, f"{split}.fasta")
        with open(path, "w") as fh:
            for seq_id, label, seq in records:
                fh.write(f">{seq_id}|{label}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i:i + 60] + "\n")
        print(f"wrote {path} ({len(records)} sequences)")

    # Unlabelled file for trying predict.py
    path = os.path.join(OUT, "unlabelled.fasta")
    with open(path, "w") as fh:
        for i in range(10):
            src = family_x if i % 2 == 0 else rng.choice(others)
            fh.write(f">query_{i + 1:02d}\n{mutate(rng, src)}\n")
    print(f"wrote {path} (10 sequences, no labels)")


if __name__ == "__main__":
    main()
