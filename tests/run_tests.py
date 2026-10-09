#!/usr/bin/env python
"""Offline test suite for the ESM-2 protein classifier.

Run from the repository root:
    python tests/run_tests.py

No internet or GPU is needed: the tests build a tiny, randomly initialised
ESM-2 (real ESM-2 vocabulary and architecture, just very small) in a temporary
folder. They check correctness of the code, not the quality of real ESM-2.
Takes about 2-4 minutes on a laptop CPU.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import (  # noqa: E402
    EsmConfig, EsmForMaskedLM, EsmForSequenceClassification, EsmTokenizer,
)

from src.config import load_config  # noqa: E402
from src.embeddings import EmbeddingStore, compute_embeddings, load_backbone, pool  # noqa: E402
from src.model import ClassificationHead, head_logits  # noqa: E402

ESM_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I",
             "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
             ".", "-", "<null_1>", "<mask>"]
DATA = os.path.join(ROOT, "example_data")
TMP = tempfile.mkdtemp(prefix="esm2clf_tests_")
TINY = os.path.join(TMP, "tiny_esm")
RESULTS = []


# ----------------------------------------------------------------- helpers
def make_tiny_esm(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "vocab.txt"), "w") as fh:
        fh.write("\n".join(ESM_VOCAB) + "\n")
    EsmTokenizer(os.path.join(path, "vocab.txt")).save_pretrained(path)
    torch.manual_seed(0)
    cfg = EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                    intermediate_size=64, max_position_embeddings=1026,
                    position_embedding_type="rotary", pad_token_id=1, mask_token_id=32,
                    token_dropout=True, emb_layer_norm_before=False,
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    with contextlib.redirect_stderr(io.StringIO()):
        EsmForMaskedLM(cfg).save_pretrained(path)


def run(script, *args, expect_fail=False):
    """Run train.py / predict.py; return combined output."""
    cmd = [sys.executable, os.path.join(ROOT, script), *args]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if expect_fail:
        assert p.returncode != 0, f"{script} should have failed but succeeded"
    elif p.returncode != 0:
        raise AssertionError(f"{script} failed:\n{out[-3000:]}")
    return out


def train(out, *extra, cache=None, val=True, expect_fail=False):
    args = ["--model", TINY, "--out", out, "--epochs", "4", "--lr", "5e-3"]
    args += ["--cache-dir", cache] if cache else ["--no-cache"]
    if not val:
        args.append("--no-val")
    return run("train.py", *args, *extra, expect_fail=expect_fail)


def load_summary(out):
    with open(os.path.join(out, "summary.json")) as fh:
        return json.load(fh)


def tmp(name):
    return os.path.join(TMP, name)


def test(fn):
    name = fn.__name__
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}\n{traceback.format_exc()}")
    return fn


def tokenizer():
    return EsmTokenizer.from_pretrained(TINY)


# ------------------------------------------------------------------- tests
def test_frozen_esm_same_in_train_and_eval_mode():
    """Justifies caching: a frozen ESM-2 gives identical outputs in both modes."""
    tok, bb = tokenizer(), load_backbone(TINY)
    batch = tok(["MKTAYIAKQRQISFVKSHFSRQ", "MSTNPKPQRKTKRNTNRRPQDVKFPGG"],
                padding=True, return_tensors="pt")
    with torch.no_grad():
        bb.eval()
        a = bb(**batch).last_hidden_state
        bb.train()
        b = bb(**batch).last_hidden_state
    assert torch.allclose(a, b, atol=1e-6), "outputs differ between train and eval mode"


def test_cls_pooling_matches_huggingface_classifier():
    """cls embedding + our head == EsmForSequenceClassification with the same weights."""
    tok = tokenizer()
    with contextlib.redirect_stderr(io.StringIO()):
        hf = EsmForSequenceClassification.from_pretrained(TINY, num_labels=3).eval()
    seqs = ["MKTAYIAKQRQISFVKSHFSRQ", "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRK",
            "MAH", "MKKLLIVSAALLAGCSQVNAETIKVGVLHSLSGTMAISETAL"]
    head = ClassificationHead(32, 3)
    head.dense.load_state_dict(hf.classifier.dense.state_dict())
    head.out_proj.load_state_dict(hf.classifier.out_proj.state_dict())
    emb = compute_embeddings(load_backbone(TINY), tok, seqs, 1024, batch_size=3)
    ours = head_logits(head, emb["cls"])
    with torch.no_grad():
        ref = torch.cat([hf(**tok([s], return_tensors="pt")).logits for s in seqs]).numpy()
    assert np.allclose(ours, ref, atol=1e-5), f"max diff {np.abs(ours - ref).max()}"


def test_mean_pooling_excludes_special_tokens_and_padding():
    tok, bb = tokenizer(), load_backbone(TINY)
    seqs = ["MKTAYIAKQR", "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGV", "MAHW"]
    emb = compute_embeddings(bb, tok, seqs, 1024, batch_size=3)    # padded batch
    for i, s in enumerate(seqs):
        with torch.no_grad():
            h = bb(**tok([s], return_tensors="pt")).last_hidden_state[0]
        manual = h[1:-1].mean(0).numpy()                             # residues only
        assert np.allclose(emb["mean"][i], manual, atol=1e-5), f"mean wrong for seq {i}"
        assert np.allclose(emb["cls"][i], h[0].numpy(), atol=1e-5), f"cls wrong for seq {i}"
    # same results whatever the batch composition (padding must not leak in)
    single = compute_embeddings(bb, tok, seqs, 1024, batch_size=1)
    for p in ("cls", "mean"):
        assert np.allclose(single[p], emb[p], atol=1e-5), f"{p} depends on batching"


def test_pool_function_masks_correctly():
    hidden = torch.arange(2 * 5 * 1, dtype=torch.float32).view(2, 5, 1)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])           # 2nd padded
    cls, mean = pool(hidden, mask)
    assert cls.flatten().tolist() == [0.0, 5.0]
    assert mean.flatten().tolist() == [2.0, 6.0]                      # (1+2+3)/3, (6)/1


def test_long_sequences_truncated_with_eos():
    tok = tokenizer()
    long_seq = "M" + "ACDEFGHIKL" * 150                                 # 1501 aa
    ids = tok([long_seq], truncation=True, max_length=1024)["input_ids"][0]
    assert len(ids) == 1024 and ids[0] == 0 and ids[-1] == 2, "truncation lost <cls>/<eos>"
    emb = compute_embeddings(load_backbone(TINY), tok, [long_seq, "MKT"], 1024)
    assert np.isfinite(emb["mean"]).all() and np.isfinite(emb["cls"]).all()


def _cli(**overrides):
    import argparse
    base = dict(train="a", test="b", val=None, model=None, max_length=None, pooling=None,
                cache_dir=None, embed_batch_size=None, epochs=None, lr=None, batch_size=None,
                selection_metric=None, seed=None, out=None, no_val=False, no_cache=False,
                save_backbone=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _yaml(name, text):
    path = tmp(name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def test_defaults_when_settings_missing_or_null():
    # no config file and nothing on the command line
    cfg = load_config(None, _cli())
    assert cfg["model"]["pooling"] == "cls" and cfg["training"]["epochs"] == 10
    # config file that does not mention pooling / epochs at all
    omitted = _yaml("omitted.yaml", "model:\n  name: x\n")
    cfg = load_config(omitted, _cli())
    assert cfg["model"]["pooling"] == "cls" and cfg["training"]["epochs"] == 10
    # config file that sets them to null explicitly
    nulls = _yaml("nulls.yaml", "model:\n  pooling: null\n  max_length: null\n"
                                "training:\n  epochs: null\n")
    cfg = load_config(nulls, _cli())
    assert cfg["model"]["pooling"] == "cls"
    assert cfg["training"]["epochs"] == 10
    assert cfg["model"]["max_length"] == 1024
    # the command line wins over the file
    cfg = load_config(nulls, _cli(pooling="mean", epochs=3))
    assert cfg["model"]["pooling"] == "mean" and cfg["training"]["epochs"] == 3


def test_yaml_false_means_not_given():
    path = _yaml("false.yaml", "data:\n  val: false\nembeddings:\n  cache_dir: false\n")
    cfg = load_config(path, _cli())
    assert cfg["data"]["val"] is None, f"val: false gave {cfg['data']['val']!r}"
    assert cfg["embeddings"]["cache_dir"] is None
    for text in ('"false"', "no", '"null"', "''"):
        cfg = load_config(_yaml("t.yaml", f"data:\n  val: {text}\n"), _cli())
        assert cfg["data"]["val"] is None, f"val: {text} not treated as empty"


def test_fp16_setting_parsed_correctly():
    from src.config import parse_fp16
    for value, expected in [(True, True), (False, False), ("auto", "auto"), ("AUTO", "auto"),
                            ("true", True), ("false", False), ("no", False), ("yes", True),
                            ("0", False), ("1", True)]:
        assert parse_fp16(value) == expected, f"{value!r} -> {parse_fp16(value)!r}"
    cfg = load_config(_yaml("fp.yaml", 'embeddings:\n  fp16: "false"\n'), _cli())
    assert cfg["embeddings"]["fp16"] is False, "quoted 'false' must turn fp16 off"
    try:
        load_config(_yaml("fp2.yaml", "embeddings:\n  fp16: maybe\n"), _cli())
        raise AssertionError("invalid fp16 value accepted")
    except ValueError as e:
        assert "fp16 must be" in str(e)


def test_precision_is_part_of_cache_key():
    from src.embeddings import cache_key, resolve_precision
    seqs = ["MKTAY", "MSTNP"]
    assert cache_key("m", 1024, seqs, "fp16") != cache_key("m", 1024, seqs, "fp32")
    assert cache_key("m", 1024, seqs, "fp32") == cache_key("m", 1024, list(seqs), "fp32")
    if not torch.cuda.is_available():                 # fp16 is never used without a GPU
        assert resolve_precision(True) == resolve_precision("auto") == "fp32"


def test_max_length_capped_at_1024():
    assert load_config(None, _cli(max_length=1024))["model"]["max_length"] == 1024
    for bad in (1025, 1026, 2):
        try:
            load_config(None, _cli(max_length=bad))
            raise AssertionError(f"max_length {bad} accepted")
        except ValueError as e:
            assert "between 3 and 1024" in str(e)


def test_stop_codons_removed_everywhere():
    from src.data import read_fasta
    path = _yaml("stops.fasta", ">a|x\nANPP*NNNN\n>b|x\nMKTAY*\n>c|y\nmkt*ay**\n>d|y\nMKTAY\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, seqs, _ = read_fasta(path)
    assert seqs == ["ANPPNNNN", "MKTAY", "MKTAY", "MKTAY"], seqs
    log = buf.getvalue()
    assert "trailing stop codon '*' from 1 sequence" in log, log
    assert "2 sequence(s) had '*' inside the sequence" in log, log


def test_train_cls_with_validation():
    out = tmp("run_cls_val")
    log = train(out, cache=tmp("cache"))
    s = load_summary(out)
    assert s["pooling"] == "cls" and s["selection_set"] == "validation"
    assert "VALIDATION set" in s["selection_note"]
    assert s["test_metrics_final_epoch"] is None
    for f in ("epoch_results.csv", "epoch_metrics.png", "training_loss.png",
              "classification_report.txt", "confusion_matrix.csv", "confusion_matrix.png",
              "predictions_test.csv", "run_config.yaml", "training_log.json",
              "best_model/head.pt", "best_model/classifier_config.json"):
        assert os.path.exists(os.path.join(out, f)), f"missing output {f}"
    er = pd.read_csv(os.path.join(out, "epoch_results.csv"))
    assert er["epoch"].tolist() == [1, 2, 3, 4], "epochs not numbered 1..N"
    assert er["selected"].sum() == 1
    assert int(er.loc[er["selected"], "epoch"].iloc[0]) == s["best_epoch"]
    assert er.loc[er["selected"], "mcc"].iloc[0] == er["mcc"].max()
    assert log.count("computing embeddings") == 3, "expected 3 sets embedded on first run"


def test_cache_reused_and_results_reproducible():
    out = tmp("run_cls_val_again")
    log = train(out, cache=tmp("cache"))
    assert "computing embeddings" not in log, "embeddings recomputed despite cache"
    assert "Loading ESM-2 backbone" not in log, "backbone loaded although all cached"
    a = pd.read_csv(os.path.join(tmp("run_cls_val"), "predictions_test.csv"))
    b = pd.read_csv(os.path.join(out, "predictions_test.csv"))
    assert a.equals(b), "same seed + same data should give identical predictions"


def test_mean_pooling_uses_same_cache():
    out = tmp("run_mean_val")
    log = train(out, "--pooling", "mean", cache=tmp("cache"))
    assert "computing embeddings" not in log, "mean pooling should reuse cached embeddings"
    s = load_summary(out)
    assert s["pooling"] == "mean"
    with open(os.path.join(out, "best_model", "classifier_config.json")) as fh:
        assert json.load(fh)["pooling"] == "mean"
    a = pd.read_csv(os.path.join(tmp("run_cls_val"), "predictions_test.csv"))
    b = pd.read_csv(os.path.join(out, "predictions_test.csv"))
    assert not np.allclose(a["confidence"], b["confidence"]), "cls and mean gave identical output"


def test_relabelled_sequences_reuse_cache():
    relabel = tmp("train_relabelled.fasta")
    with open(os.path.join(DATA, "train.fasta")) as src, open(relabel, "w") as dst:
        for line in src:
            dst.write(line.replace("|familyX", "|groupA").replace("|other", "|groupB"))
    for name in ("val", "test"):
        with open(os.path.join(DATA, f"{name}.fasta")) as src, \
                open(tmp(f"{name}_relabelled.fasta"), "w") as dst:
            for line in src:
                dst.write(line.replace("|familyX", "|groupA").replace("|other", "|groupB"))
    log = train(tmp("run_relabel"), "--train", relabel, "--val", tmp("val_relabelled.fasta"),
                "--test", tmp("test_relabelled.fasta"), cache=tmp("cache"))
    assert "computing embeddings" not in log, "new labels on the same sequences should hit cache"
    assert load_summary(tmp("run_relabel"))["classes"] == ["groupA", "groupB"]


def test_changing_max_length_recomputes():
    log = train(tmp("run_maxlen"), "--max-length", "100", cache=tmp("cache"))
    assert log.count("computing embeddings") == 3, "different max_length must not reuse cache"
    assert "truncated" in log, "expected a truncation note with max_length 100"


def test_train_without_validation():
    out = tmp("run_noval")
    train(out, val=False)
    s = load_summary(out)
    assert s["selection_set"] == "test" and "TEST set" in s["selection_note"]
    assert s["test_metrics_final_epoch"] is not None
    assert set(s["test_metrics_best_epoch"]) == {
        "mcc", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted"}
    er = pd.read_csv(os.path.join(out, "epoch_results.csv"))
    best = er.loc[er["selected"]].iloc[0]
    assert abs(best["mcc"] - s["test_metrics_best_epoch"]["mcc"]) < 1e-12


def _snapshot(folder):
    """Every file under ``folder`` with its size and modification time."""
    found = {}
    for dirpath, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            found[p] = (st.st_size, st.st_mtime_ns)
    return found


def test_no_cache_writes_nothing():
    # train.py runs with the repository as working directory, so a broken
    # --no-cache would write to the default cache folder there.
    repo_cache = os.path.join(ROOT, "embedding_cache")
    before_repo, before_tmp = _snapshot(repo_cache), set(os.listdir(TMP))
    train(tmp("run_nocache"), val=False)
    assert _snapshot(repo_cache) == before_repo, f"--no-cache wrote to {repo_cache}"
    new = set(os.listdir(TMP)) - before_tmp - {"run_nocache"}
    assert not new, f"unexpected files created: {new}"


def _check_predict_matches(run_dir):
    out_csv = os.path.join(run_dir, "repredict.csv")
    log = run("predict.py", "--model", os.path.join(run_dir, "best_model"),
              "--fasta", os.path.join(DATA, "test.fasta"), "--out", out_csv)
    a = pd.read_csv(os.path.join(run_dir, "predictions_test.csv"))
    b = pd.read_csv(out_csv)
    assert (a["predicted_label"] == b["predicted_label"]).all(), "predict.py disagrees with train.py"
    assert np.allclose(a["confidence"], b["confidence"], atol=1e-5)
    assert "Metrics against the labels" in log


def test_predict_matches_training_cls():
    _check_predict_matches(tmp("run_cls_val"))


def test_predict_matches_training_mean():
    _check_predict_matches(tmp("run_mean_val"))


def test_predict_records_and_follows_training_precision():
    run_dir = tmp("run_cls_val")
    with open(os.path.join(run_dir, "best_model", "classifier_config.json")) as fh:
        trained = json.load(fh)["embedding_precision"]
    assert trained in ("fp16", "fp32")
    assert load_summary(run_dir)["embedding_precision"] == trained
    log = run("predict.py", "--model", os.path.join(run_dir, "best_model"),
              "--fasta", os.path.join(DATA, "unlabelled.fasta"), "--out", tmp("p_prec.csv"))
    assert f"Embedding precision: {trained} (training used {trained})" in log, log[-800:]
    log = run("predict.py", "--model", os.path.join(run_dir, "best_model"), "--no-fp16",
              "--fasta", os.path.join(DATA, "unlabelled.fasta"), "--out", tmp("p_prec2.csv"))
    assert "Embedding precision: fp32" in log


def test_predict_unlabelled():
    out_csv = tmp("unlabelled_pred.csv")
    log = run("predict.py", "--model", os.path.join(tmp("run_mean_val"), "best_model"),
              "--fasta", os.path.join(DATA, "unlabelled.fasta"), "--out", out_csv)
    df = pd.read_csv(out_csv)
    assert len(df) == 10 and "true_label" not in df.columns
    assert np.allclose(df[[c for c in df.columns if c.startswith("prob_")]].sum(axis=1), 1.0)
    assert "pooling: mean" in log, "predict.py should report the pooling used in training"


def test_save_backbone_works_offline():
    out = tmp("run_backbone")
    train(out, "--save-backbone", val=False)
    assert os.path.isdir(os.path.join(out, "best_model", "backbone"))
    hidden = TINY + "_moved"                     # make the original model unavailable
    os.rename(TINY, hidden)
    try:
        _check_predict_matches(out)
    finally:
        os.rename(hidden, TINY)


def test_error_missing_label():
    bad = tmp("nolabel.fasta")
    with open(bad, "w") as fh:
        fh.write(">seq_without_label\nMKTAYIAKQR\n")
    log = train(tmp("run_err1"), "--test", bad, expect_fail=True)
    assert "have no label in the header" in log


def test_error_unseen_label():
    bad = tmp("unseen.fasta")
    shutil.copy(os.path.join(DATA, "test.fasta"), bad)
    with open(bad, "a") as fh:
        fh.write(">weird1|newclass\nMKTAYIAKQR\n")
    log = train(tmp("run_err2"), "--test", bad, expect_fail=True)
    assert "never appear in the training set: ['newclass']" in log


def test_error_bad_options():
    log = run("train.py", "--model", TINY, "--epochs", "0", expect_fail=True)
    assert "epochs must be at least 1" in log
    log = run("train.py", "--model", TINY, "--pooling", "max", expect_fail=True)
    assert "invalid choice" in log
    log = run("train.py", "--model", TINY, "--selection-metric", "auc", expect_fail=True)
    assert "selection_metric must be one of" in log
    log = run("train.py", "--model", TINY, "--max-length", "1026", expect_fail=True)
    assert "between 3 and 1024" in log


def main():
    print(f"Building a tiny offline ESM-2 in {TMP}")
    make_tiny_esm(TINY)
    print("Running tests ...")
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        test(fn)
    shutil.rmtree(TMP, ignore_errors=True)
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
