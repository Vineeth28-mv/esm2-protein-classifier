"""Compute frozen ESM-2 protein embeddings once and cache them on disk.

For every sequence two protein-level vectors are computed in the same pass:
  * cls  - the vector of the <cls> start token (what Hugging Face's
           EsmForSequenceClassification uses)
  * mean - the average of the residue vectors (start/end tokens and
           padding excluded), the standard ESM per-protein embedding
Both are cached, so switching the pooling method needs no recomputation.

The cache key depends on the model, max_length, the numeric precision (fp16
on a GPU or fp32) and the sequences themselves - not on labels - so
re-labelling the same sequences, or changing any training setting, reuses the
cached embeddings, while fp16 and fp32 embeddings never get mixed.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import EsmModel

CACHE_VERSION = 2
POOLINGS = ("cls", "mean")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(fp16):
    """Precision actually used: "fp16" only when requested AND a GPU is present.

    ``fp16`` may be "auto" (fp16 whenever a GPU is available), True or False.
    """
    on_gpu = torch.cuda.is_available()
    if fp16 == "auto":
        return "fp16" if on_gpu else "fp32"
    return "fp16" if (fp16 and on_gpu) else "fp32"


def load_backbone(name):
    """ESM-2 encoder without any task head, frozen and in eval mode."""
    model = EsmModel.from_pretrained(name, add_pooling_layer=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def pool(hidden, attention_mask):
    """Return (cls, mean) vectors from last-layer hidden states.

    hidden:          (batch, tokens, hidden_size)
    attention_mask:  (batch, tokens), 1 for real tokens incl. <cls> and <eos>
    """
    cls = hidden[:, 0, :]
    residue_mask = attention_mask.clone()
    residue_mask[:, 0] = 0                                    # drop <cls>
    eos_pos = attention_mask.sum(dim=1) - 1                   # last real token is <eos>
    residue_mask[torch.arange(hidden.size(0), device=hidden.device), eos_pos] = 0
    residue_mask = residue_mask.unsqueeze(-1).to(hidden.dtype)
    counts = residue_mask.sum(dim=1).clamp(min=1.0)
    mean = (hidden * residue_mask).sum(dim=1) / counts
    return cls, mean


@torch.no_grad()
def compute_embeddings(backbone, tokenizer, seqs, max_length, batch_size=32, fp16=False,
                       desc="sequences"):
    """Embed sequences with a frozen ESM-2. Returns {"cls": (N,H), "mean": (N,H)} float32."""
    device = get_device()
    backbone.to(device)
    backbone.eval()

    enc = tokenizer(list(seqs), truncation=True, padding=False, max_length=max_length,
                    add_special_tokens=True)
    input_ids = enc["input_ids"]
    n = len(input_ids)
    hidden_size = backbone.config.hidden_size
    out = {p: np.zeros((n, hidden_size), dtype=np.float32) for p in POOLINGS}

    # Longest first: batches of similar length need little padding, and an
    # out-of-memory error (if any) shows up immediately rather than at the end.
    order = sorted(range(n), key=lambda i: -len(input_ids[i]))
    use_amp = fp16 and device.type == "cuda"
    t0 = time.time()
    next_report = 0.1
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer.pad({"input_ids": [input_ids[i] for i in idx]}, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            hidden = backbone(**batch).last_hidden_state
        cls, mean = pool(hidden.float(), batch["attention_mask"])
        out["cls"][idx] = cls.cpu().numpy()
        out["mean"][idx] = mean.cpu().numpy()

        done = min(start + batch_size, n) / n
        if n >= 1000 and done >= next_report:
            print(f"    {desc}: {100 * done:5.1f}% ({time.time() - t0:,.0f} s)")
            next_report += 0.1
    return out


def cache_key(model_name, max_length, seqs, precision="fp32"):
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}|{model_name}|{max_length}|{precision}\n".encode())
    for s in seqs:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:24]


class EmbeddingStore:
    """Loads embeddings from the cache, or computes (and caches) them.

    The ESM-2 backbone is only loaded if something actually needs computing,
    so a fully cached re-run starts training within seconds.
    """

    def __init__(self, model_name, tokenizer, max_length, cache_dir=None, batch_size=32,
                 fp16="auto", backbone_source=None):
        """``backbone_source``: where to load ESM-2 from if it differs from
        ``model_name`` (e.g. a copy saved inside a model folder). The cache
        key always uses ``model_name``, so caches are shared between both."""
        self.model_name = model_name
        self.backbone_source = backbone_source or model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.precision = resolve_precision(fp16)     # "fp16" or "fp32"
        self._backbone = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @property
    def backbone(self):
        if self._backbone is None:
            print(f"  Loading ESM-2 backbone: {self.backbone_source}")
            self._backbone = load_backbone(self.backbone_source)
        return self._backbone

    def get(self, seqs, set_name):
        """Return ({"cls": array, "mean": array}, from_cache: bool)."""
        path = None
        if self.cache_dir:
            key = cache_key(self.model_name, self.max_length, seqs, self.precision)
            path = os.path.join(self.cache_dir, f"{key}.npz")
            if os.path.exists(path):
                with np.load(path) as data:
                    emb = {p: data[p] for p in POOLINGS}
                if all(len(emb[p]) == len(seqs) for p in POOLINGS):
                    print(f"  {set_name}: loaded {len(seqs):,} embeddings from cache ({path})")
                    return emb, True
                print(f"  {set_name}: cache file has the wrong size; recomputing.")

        print(f"  {set_name}: computing embeddings for {len(seqs):,} sequences ...")
        t0 = time.time()
        emb = compute_embeddings(self.backbone, self.tokenizer, seqs, self.max_length,
                                 batch_size=self.batch_size,
                                 fp16=self.precision == "fp16", desc=set_name)
        print(f"  {set_name}: done in {time.time() - t0:,.1f} s")

        if path:
            tmp = path + ".tmp.npz"
            np.savez(tmp, **emb)
            os.replace(tmp, path)
            meta = {"model": self.model_name, "max_length": self.max_length,
                    "precision": self.precision, "n_sequences": len(seqs), "set": set_name,
                    "cache_version": CACHE_VERSION}
            with open(os.path.splitext(path)[0] + ".json", "w") as fh:
                json.dump(meta, fh, indent=1)
            print(f"  {set_name}: cached to {path}")
        return emb, False
