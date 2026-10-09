# Interpreting the results

## The metrics

Which set the numbers come from:

| Output | Computed on |
|---|---|
| `summary.json`, `classification_report.txt`, `confusion_matrix.*`, `predictions_test.csv` | the **test set**, using the selected epoch |
| `epoch_results.csv`, `epoch_metrics.png` | the set used to **choose** the epoch: the **validation set** if you gave one, otherwise the test set (the `evaluated_on` column says which) |

| Metric | Range | What it measures |
|---|---|---|
| **MCC** (Matthews correlation coefficient) | −1 to 1 | overall agreement between predictions and truth, robust to class imbalance; **the headline number** |
| Accuracy | 0 to 1 | fraction of sequences predicted correctly; **misleading when classes are imbalanced** |
| Precision | 0 to 1 | of the sequences predicted as a class, how many really are |
| Recall | 0 to 1 | of the sequences that are a class, how many were found |
| F1 | 0 to 1 | balance of precision and recall |

### Macro vs. weighted averages

Precision, recall and F1 are first computed **per class** (see `classification_report.txt`), then combined into one number in two ways. Both are reported (`*_macro` and `*_weighted` columns).

| Average | How it is combined | Use it for |
|---|---|---|
| **macro** | plain average over classes; every class counts equally | **the headline, with MCC**: shows whether the model works on *every* class, including small ones |
| **weighted** | average weighted by the number of test sequences per class | performance on a typical sequence; usually close to accuracy |

When the two differ a lot (e.g. macro F1 = 0.62, weighted F1 = 0.91), the model does well on large classes but poorly on some small ones; check the per-class report to see which. A "micro" average is not reported because, for this type of classification, it is identical to accuracy.

### Why MCC and not accuracy?

Suppose 95% of your test set is `other`. A model that **always** predicts `other` has 95% accuracy, yet it is useless. Its MCC is 0. MCC only gets high when the model is right about every class.

Rough guide for MCC:

| MCC | Interpretation |
|---|---|
| > 0.9 | excellent separation |
| 0.7 – 0.9 | good |
| 0.4 – 0.7 | partial signal; inspect the errors |
| < 0.4 | weak; little useful signal |
| ≈ 0 | no better than chance |

These are only rough guides. Whether 0.7 is good depends on how hard the task is (see the negative set discussion in [data_format.md](data_format.md)).

## The per-class report (`classification_report.txt`)

```text
              precision    recall  f1-score   support
     familyX     0.9302    1.0000    0.9639        40
       other     1.0000    0.9250    0.9610        40
```

- `support` is the number of test sequences of that class. Metrics for classes with small support are noisy.
- **Low recall** for a class means the model **misses** its members (false negatives).
- **Low precision** means the model **over-predicts** it (false positives).

## The confusion matrix

Rows are the **true** class, columns the **predicted** class. The diagonal holds the correct predictions. The PNG colours each row by the fraction of that true class (that is, recall), with raw counts written in the cells.

Look for:

- **Off-diagonal blocks between specific classes:** these classes look alike to the model. They are often biologically related, which is interesting in itself.
- **A column that collects many errors:** the model uses that class as a default. This often happens with a large or heterogeneous `other` class.

## The epoch plot (`epoch_metrics.png`, `epoch_results.csv`)

- **Rising, then flat:** normal; training has converged.
- **Rising, then falling:** overfitting after the peak. This is the case where picking the best epoch matters.
- **Still rising at the last epoch:** train longer (`--epochs 20`).
- **Noisy, with big jumps between epochs:** the evaluation set is probably small; differences between epochs may be chance.

Without a validation set, `summary.json` also shows the **final epoch's** test scores. If they are close to the best epoch's, epoch selection has had little effect on the reported numbers.

## The training loss (`training_loss.png`)

The loss should go down and level off. If it stays flat from the start, the learning rate may be too low, or the classes may not be separable with this model. If it is very erratic, try a lower learning rate (`--lr 5e-4`) or a larger batch size.

## Per-sequence predictions (`predictions_test.csv`)

This is often the most informative file. Useful things to do:

1. **Sort the mistakes by confidence.** High-confidence errors are frequently **mislabelled sequences** or genuinely ambiguous ones (multi-domain proteins, fragments). Checking ten of them by hand teaches you a lot.
2. **Look at low-confidence correct predictions**, i.e. borderline cases.
3. **Plot confidence for correct vs. incorrect predictions.** If mistakes mostly have low confidence, a confidence threshold can trade recall for precision.

Remember that confidences are **not calibrated** probabilities. Use them to rank, not as literal probabilities.

## Warning signs

| What you see | Likely cause |
|---|---|
| Near-perfect scores on an "interesting" task | similarity leakage between train and test (cluster and re-split), or negatives that are too easy |
| MCC ≈ 0, model predicts one class for everything | learning rate too low or too few epochs; classes not separable by this model; check the class weights in the log |
| Validation scores much higher than test scores | validation and test sets differ in composition; check class distributions |
| Many sequences truncated (note in the log) | proteins longer than 1022 aa; the model only sees the first 1022 residues |
| Big differences between runs with different `--seed` | small dataset; treat differences as noise and report the mean over seeds |

## Comparing experiments

To answer "is ESM-2 good at capturing family X?", it helps to compare:

- **model sizes** (8M vs 35M vs 150M): does a bigger model capture more?
- **similarity thresholds** (80% vs 30% clustering): does the signal survive at low identity?
- **a simple baseline**, e.g. BLAST or HMMER best hit against the training set: does ESM-2 add anything beyond sequence similarity?

The last comparison is the most convincing evidence that the language model captures something beyond plain sequence similarity.
