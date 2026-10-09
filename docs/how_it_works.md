# How it works

This page explains, without assuming a machine-learning background, what the pipeline does and what you can expect from it.

## 1. ESM-2: a language model for proteins

ESM-2 is a neural network trained by Meta AI on tens of millions of natural protein sequences (UniRef50). It was trained on one task: hide some amino acids in a sequence and predict them from the rest. To get good at this across all of protein space, it had to learn which residues co-vary, which motifs belong together, what is conserved, and much about structure and function. It was never told any of this explicitly.

After that training, every protein you feed in comes out as a list of numbers, a **representation** or **embedding**. Proteins that are similar in ways ESM-2 has learned to recognise end up with similar representations, even when their sequences share little identity.

## 2. From a protein to one vector

ESM-2 produces one vector per position: one for each amino acid, plus one for the special `<cls>` token added at the start and one for `<eos>` at the end. To classify a whole protein we need **one vector per protein**. There are two common ways to get it, chosen with `pooling`:

```
 <cls>   M     K     K     L     L    ...   <eos>
   ↓     ↓     ↓     ↓     ↓     ↓            ↓
  v0    v1    v2    v3    v4    v5    ...    vN

 cls  pooling:  protein vector = v0
 mean pooling:  protein vector = average(v1 ... vN-1)     (residues only)
```

- **`cls` (default).** Uses the `<cls>` vector, as Hugging Face's `EsmForSequenceClassification` does and as the original mobileOG pipeline did. Through ESM-2's attention layers this token has "seen" the whole sequence. However, ESM-2 was trained **only** to predict hidden amino acids, never to summarise a protein in the `<cls>` position (unlike BERT, whose `<cls>` token had its own training task). With a frozen ESM-2 the `<cls>` vector stays exactly as pretraining left it.
- **`mean`.** Averages the vectors of the real amino acids, excluding `<cls>`, `<eos>` and padding. Every residue vector *was* trained during pretraining, so the average reflects what ESM-2 learned about the whole sequence. This is the per-protein embedding produced by Meta's own ESM extraction script and the usual choice in work with frozen ESM embeddings. One study on ESM-2 found `<cls>` vectors of proteins from different folds to be nearly identical (cosine similarity > 0.9), whereas mean-pooled vectors varied much more, and found mean pooling the strongest of the simple pooling baselines.

The evidence favours `mean` for a frozen ESM-2, but the size of the difference depends on the task, so the best approach is to try both (the cache makes this nearly free). The default remains `cls` so the pipeline reproduces the original method exactly.

```
protein sequence  ──►  ESM-2 (frozen)  ──►  one vector  ──►  classification head  ──►  class probabilities
  MKKLLIVSA...         8M–650M params       320–1280 numbers    small, trainable         familyX: 0.97
                       run once, cached     (cls or mean)                                other:   0.03
```

## 3. Frozen backbone + classification head

The pipeline **freezes** ESM-2: its weights never change. Only the **classification head** is trained. This is a small two-layer network (`dense` → `tanh` → `out_proj`, identical in layout to Hugging Face's ESM classification head) that maps the protein vector to one score per class.

### Why embeddings can be computed once

A frozen ESM-2 always gives the same output for the same sequence; it even behaves identically in training and evaluation mode, since ESM-2 uses no dropout. Running ESM-2 inside every training step (as a standard Hugging Face `Trainer` setup does) therefore recomputes the same vectors in every epoch. This pipeline instead computes each protein's vectors **once**, stores them in the embedding cache, and trains the head on the stored vectors. The resulting model is the same; training is much faster, and later runs on the same sequences skip ESM-2 entirely. The test suite checks that `cls` pooling with this head gives exactly the same predictions as Hugging Face's `EsmForSequenceClassification` with the same weights.

Why freeze?

- **It is fast and cheap.** Only a tiny fraction of parameters is trained: about 1.3% for the 8M model (roughly 103,000 of 7.8 million), and less for bigger models (about 0.25% for 650M).
- **It needs little data.** With few trainable parameters, it is much harder to overfit small datasets.
- **It directly tests the question "does ESM-2 already capture this?"** If a small head on frozen representations separates your classes well, the information is already in ESM-2's representation of those proteins.

The trade-off: if the distinction you care about is subtle and not well represented in ESM-2's general-purpose embedding, a frozen model will plateau. Fine-tuning ESM-2 itself (not included here) can then help, but it needs more data, more GPU memory and more care to avoid overfitting.

## 4. Training details

- **Loss:** cross-entropy, **weighted by class frequency** (`class_weighting: balanced`). A class with 10× fewer examples gets about 10× more weight per example, so the model cannot score well by ignoring it.
- **Optimizer:** the same optimisation method as the Hugging Face `Trainer` (AdamW, linear learning-rate decay to 0 without warmup, gradient clipping at 1.0, no weight decay on biases), with the settings of the original mobileOG pipeline: learning rate `1e-3`, weight decay `0.01` and batch size 16. (These are not the `Trainer`'s own defaults, which are 5e-5, 0.0 and 8.)
- **Epochs:** after every epoch the head is evaluated, and the best epoch (by MCC by default) is kept. See the README for how validation and test sets are used.
- **Length:** ESM-2 handles up to 1024 tokens, i.e. 1022 amino acids plus start and end tokens. Longer proteins are truncated to their first 1022 residues, and a note says how many were affected.
- **Embedding computation:** sequences are sorted by length and padded per batch, which keeps memory use and run time low. On a GPU, ESM-2 runs in half precision (fp16) by default (`embeddings.fp16`); on a CPU it always uses full precision (fp32). The precision is part of the cache key and is saved with the model, so `predict.py` uses the same one by default.
- **Stop codons:** ESM-2 has no token for `*`, so every `*` is removed before tokenisation, wherever it occurs (`ANPP*NNNN` → `ANPPNNNN`). The log reports how many sequences had a trailing `*` and lists sequences with an internal `*`, which can indicate pseudogenes or frameshifts.
- **Reproducibility:** with the same data, settings and seed, results are identical from run to run.

## 5. What is feasible?

Rough expectations from experience with protein language models:

| Task | Typical outcome with a frozen ESM-2 head |
|---|---|
| Distinguishing well-defined protein families or functional classes | usually works well, often MCC > 0.8 |
| Recognising family members at low sequence identity (< 30%) to training examples | often works noticeably better than BLAST-style similarity, which is a main attraction of PLMs |
| Fine distinctions within a family (e.g. substrate specificity, subfamilies) | variable; larger models (150M/650M) help; may need fine-tuning |
| Properties that depend on context outside the protein (genomic neighbourhood, host, expression) | limited; the model only sees the sequence |
| Very small classes (tens of examples) | possible for distinct families, but metrics become noisy; use a validation set and treat results as exploratory |

The best way to find out is to run it: start with the 8M model, check the results, then see whether `mean` pooling or a larger model improves them.

## 6. Limitations to keep in mind

- **Similarity leakage** is the biggest risk. If test proteins are close homologs of training proteins, high scores may only reflect sequence similarity. Always split by clusters (see [data_format.md](data_format.md)).
- **The negative set defines the task.** "familyX vs. random proteins" is much easier than "familyX vs. its closest relatives". Choose negatives that match the question you want to answer.
- **Probabilities are not calibrated.** A confidence of 0.95 does not mean "95% chance of being correct". Use confidences for ranking, and check precision at a threshold on held-out data if you need one.
- **Frozen means general-purpose.** The representation was not tailored to your task. If performance plateaus, that is informative in itself.

## References

- ESM-2: Lin Z, *et al.* Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379: 1123–1130 (2023).
- ESM embedding extraction (`--include mean` / `bos`): the `scripts/extract.py` script in Meta's [esm repository](https://github.com/facebookresearch/esm).
- Comparison of `<cls>` and mean pooling on ESM-2 (35M–650M): Locality-aware pooling enhances protein language model performance across varied applications, [PMC12261463](https://pmc.ncbi.nlm.nih.gov/articles/PMC12261463).
- Mean pooling as the strongest simple baseline on ESM-2 650M: Pool PaRTI, [bioRxiv 2024.10.04.616701](https://www.biorxiv.org/content/10.1101/2024.10.04.616701).
