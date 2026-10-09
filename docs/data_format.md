# Data format

## Training files

Each header holds a **sequence ID** and a **label**, joined by `|`:

```text
>WP_0001|familyX
MKKLLIVSAALLAGCSQVNAETIKVGVLHS...
>WP_0002|other
MSNQEIRVLLVDDHPLVRRGLEQLL...
```

```bash
python train.py --train train.fasta --val val.fasta --test test.fasta --out results/run
```

## Other header formats

If your headers look different, don't edit the file; describe the format with two flags:

- `--label-delimiter`: the separator (default `|`)
- `--label-field`: which part is the label, counting from 0 (default `1`)

| Header | Add |
|---|---|
| `>WP_0001\|familyX` | nothing |
| `>familyX\|WP_0001` | `--label-field 0` |
| `>WP_0001;familyX` | `--label-delimiter ";"` |
| `>familyX;WP_0001` | `--label-delimiter ";" --label-field 0` |
| `>sp\|P0A7V8\|RS4_ECOLI\|other` | `--label-field 3` |

Example:

```bash
python train.py --train train.fasta --test test.fasta --out results/run \
                --label-delimiter ";" --label-field 0
```

Always quote the separator (`";"`, not `;`).

## Rules

- Only the text before the first space is read: `>WP_0001|familyX some description` works, `>WP_0001 familyX` does not.
- Every header needs a label.
- Labels are case-sensitive and can't contain spaces or the separator.
- You need at least 2 classes. Every class in the validation and test files must also appear in the training file.
- Train, validation and test files must use the same header format.

## Prediction

**New sequences (true class unknown).** Any header works; the header up to the first space becomes the ID.

```text
>query_01
MKKLLIVSAALLAGCSQ...
>sp|P0A7V8|RS4_ECOLI 30S ribosomal protein S4
MARYLGPKLKLSRREGT...
```

```bash
python predict.py --model results/run/best_model --fasta new.fasta --out predictions.csv
```

| seq_id | predicted_label | confidence | prob_familyX | prob_other |
|---|---|---|---|---|
| query_01 | familyX | 0.97 | 0.97 | 0.03 |

**Sequences with known classes (to check accuracy).** Use headers with the true label, as in training, and add `--labelled`:

```bash
python predict.py --model results/run/best_model --fasta test.fasta --out predictions.csv --labelled
```

| seq_id | true_label | predicted_label | correct | confidence | prob_familyX | prob_other |
|---|---|---|---|---|---|---|
| WP_0001 | familyX | familyX | True | 0.97 | 0.97 | 0.03 |

It also prints MCC, accuracy, precision, recall and F1.

`--labelled` reuses the header format from training. If the file uses a different format, add `--label-delimiter` / `--label-field` as in the table above.
