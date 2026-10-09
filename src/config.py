"""Load config.yaml and merge command-line overrides."""

import copy

import yaml

DEFAULTS = {
    "data": {
        "train": None,
        "test": None,
        "val": None,
        "label_delimiter": "|",
        "label_field": 1,
    },
    "model": {
        "name": "facebook/esm2_t6_8M_UR50D",
        "max_length": 1024,
        "pooling": "cls",
    },
    "embeddings": {
        "cache_dir": "embedding_cache",
        "batch_size": 32,
        "fp16": "auto",
    },
    "training": {
        "epochs": 10,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "weight_decay": 0.01,
        "class_weighting": "balanced",
        "selection_metric": "mcc",
        "seed": 42,
        "logging_steps": 100,
    },
    "output": {
        "dir": "results/run",
        "save_backbone": False,
    },
}

# command-line flag name -> (section, key)
CLI_MAP = {
    "train": ("data", "train"),
    "test": ("data", "test"),
    "val": ("data", "val"),
    "model": ("model", "name"),
    "max_length": ("model", "max_length"),
    "pooling": ("model", "pooling"),
    "cache_dir": ("embeddings", "cache_dir"),
    "embed_batch_size": ("embeddings", "batch_size"),
    "epochs": ("training", "epochs"),
    "lr": ("training", "learning_rate"),
    "batch_size": ("training", "batch_size"),
    "selection_metric": ("training", "selection_metric"),
    "seed": ("training", "seed"),
    "out": ("output", "dir"),
}

POOLINGS = ("cls", "mean")

VALID_METRICS = (
    "mcc", "accuracy",
    "f1", "precision", "recall",                       # short names = macro
    "f1_macro", "precision_macro", "recall_macro",
    "f1_weighted", "precision_weighted", "recall_weighted",
)

# Text values that mean "not given" (YAML's own null/false are handled too).
_NULL_TEXT = ("", "null", "none", "false", "no", "off")
_TRUE_TEXT = ("true", "yes", "on", "1")
_FALSE_TEXT = ("false", "no", "off", "0")

# ESM-2 was trained on at most 1024 tokens (1022 residues + <cls> + <eos>).
MAX_TOKENS = 1024


def is_unset(value):
    """True for None, False (YAML `false`) and text such as "null" or "false"."""
    if value is None or value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in _NULL_TEXT


def parse_fp16(value):
    """Return "auto", True or False for the fp16 setting.

    Accepts YAML booleans and text ("auto", "true"/"false", "yes"/"no",
    "1"/"0"). Anything else is an error rather than a silent guess.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text in _TRUE_TEXT:
        return True
    if text in _FALSE_TEXT:
        return False
    raise ValueError(f"fp16 must be auto, true or false; got '{value}'.")


def _deep_update(base, new):
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, cli_args=None):
    """Return a config dict: built-in defaults <- YAML file <- command-line flags."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})

    if cli_args is not None:
        for flag, (section, key) in CLI_MAP.items():
            value = getattr(cli_args, flag, None)
            if value is not None:
                cfg[section][key] = value
        if getattr(cli_args, "no_val", False):
            cfg["data"]["val"] = None
        if getattr(cli_args, "no_cache", False):
            cfg["embeddings"]["cache_dir"] = None
        if getattr(cli_args, "save_backbone", False):
            cfg["output"]["save_backbone"] = True

    for section, key in (("data", "val"), ("embeddings", "cache_dir")):
        if is_unset(cfg[section][key]):
            cfg[section][key] = None

    _validate(cfg)
    return cfg


def _validate(cfg):
    if not cfg["data"]["train"]:
        raise ValueError("No training file given (data.train in config or --train).")
    if not cfg["data"]["test"]:
        raise ValueError("No test file given (data.test in config or --test).")
    t = cfg["training"]
    for key in ("epochs", "pooling"):
        sect = "training" if key == "epochs" else "model"
        if cfg[sect][key] is None:
            cfg[sect][key] = DEFAULTS[sect][key]
    if int(t["epochs"]) < 1:
        raise ValueError(f"epochs must be at least 1, got {t['epochs']}.")
    cfg["model"]["pooling"] = str(cfg["model"]["pooling"]).lower()
    if cfg["model"]["pooling"] not in POOLINGS:
        raise ValueError(f"pooling must be 'cls' or 'mean', got '{cfg['model']['pooling']}'.")
    if t["selection_metric"] not in VALID_METRICS:
        raise ValueError(f"selection_metric must be one of {VALID_METRICS}, "
                         f"got '{t['selection_metric']}'.")
    if t["class_weighting"] not in ("balanced", "none"):
        raise ValueError("class_weighting must be 'balanced' or 'none'.")
    if cfg["model"]["max_length"] is None:
        cfg["model"]["max_length"] = DEFAULTS["model"]["max_length"]
    if not 3 <= int(cfg["model"]["max_length"]) <= MAX_TOKENS:
        raise ValueError(f"model.max_length must be between 3 and {MAX_TOKENS} (ESM-2 was "
                         f"trained on at most {MAX_TOKENS} tokens), got "
                         f"{cfg['model']['max_length']}.")
    cfg["embeddings"]["fp16"] = parse_fp16(cfg["embeddings"]["fp16"])
    if int(t["batch_size"]) < 1 or int(cfg["embeddings"]["batch_size"]) < 1:
        raise ValueError("batch sizes must be at least 1.")
