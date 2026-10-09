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
    "label_delimiter": ("data", "label_delimiter"),
    "label_field": ("data", "label_field"),
}

# Settings where null has a meaning of its own ("none"/"disabled"); every
# other setting falls back to its default when it is null.
NULL_MEANS_NONE = {("data", "train"), ("data", "test"), ("data", "val"),
                   ("embeddings", "cache_dir")}

# Settings that must be numbers: (section, key) -> (type, smallest allowed)
NUMERIC = {
    ("model", "max_length"): (int, None),          # range checked separately
    ("embeddings", "batch_size"): (int, 1),
    ("training", "epochs"): (int, 1),
    ("training", "learning_rate"): (float, None),
    ("training", "batch_size"): (int, 1),
    ("training", "weight_decay"): (float, 0.0),
    ("training", "seed"): (int, None),
    ("training", "logging_steps"): (int, 1),
    ("data", "label_field"): (int, 0),
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


def _warn_unknown_keys(user_cfg, path):
    for section, values in (user_cfg or {}).items():
        if section not in DEFAULTS:
            print(f"WARNING: unknown section '{section}' in {path} is ignored.")
        elif isinstance(values, dict):
            for key in values:
                if key not in DEFAULTS[section]:
                    print(f"WARNING: unknown setting '{section}.{key}' in {path} is ignored "
                          f"(valid: {', '.join(DEFAULTS[section])}).")


def load_config(path=None, cli_args=None):
    """Return a config dict: built-in defaults <- YAML file <- command-line flags."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as fh:
            user_cfg = yaml.safe_load(fh) or {}
        _warn_unknown_keys(user_cfg, path)
        _deep_update(cfg, user_cfg)

    if cli_args is not None:
        if getattr(cli_args, "val", None) and getattr(cli_args, "no_val", False):
            raise ValueError("Use either --val or --no-val, not both.")
        if getattr(cli_args, "cache_dir", None) and getattr(cli_args, "no_cache", False):
            raise ValueError("Use either --cache-dir or --no-cache, not both.")
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
    # 1. Any setting left null falls back to its built-in default.
    for section, defaults in DEFAULTS.items():
        for key, default in defaults.items():
            if cfg[section].get(key) is None and (section, key) not in NULL_MEANS_NONE:
                cfg[section][key] = default

    if not cfg["data"]["train"]:
        raise ValueError("No training file given (data.train in config or --train).")
    if not cfg["data"]["test"]:
        raise ValueError("No test file given (data.test in config or --test).")

    # 2. Numbers must be numbers, within range.
    for (section, key), (kind, minimum) in NUMERIC.items():
        value = cfg[section][key]
        try:
            if isinstance(value, bool):
                raise TypeError
            converted = kind(value)
            if kind is int and float(value) != converted:
                raise TypeError
        except (TypeError, ValueError):
            raise ValueError(f"{section}.{key} must be a "
                             f"{'whole number' if kind is int else 'number'}, got '{value}'.")
        if minimum is not None and converted < minimum:
            raise ValueError(f"{key} must be at least {minimum}, got {value}.")
        cfg[section][key] = converted
    if not 3 <= cfg["model"]["max_length"] <= MAX_TOKENS:
        raise ValueError(f"model.max_length must be between 3 and {MAX_TOKENS} (ESM-2 was "
                         f"trained on at most {MAX_TOKENS} tokens), got "
                         f"{cfg['model']['max_length']}.")

    # 3. Text choices are case-insensitive.
    m, t = cfg["model"], cfg["training"]
    m["pooling"] = str(m["pooling"]).strip().lower()
    if m["pooling"] not in POOLINGS:
        raise ValueError(f"pooling must be 'cls' or 'mean', got '{m['pooling']}'.")
    t["selection_metric"] = str(t["selection_metric"]).strip().lower()
    if t["selection_metric"] not in VALID_METRICS:
        raise ValueError(f"selection_metric must be one of {', '.join(VALID_METRICS)}; "
                         f"got '{t['selection_metric']}'.")
    t["class_weighting"] = str(t["class_weighting"]).strip().lower()
    if t["class_weighting"] not in ("balanced", "none"):
        raise ValueError(f"class_weighting must be 'balanced' or 'none', "
                         f"got '{t['class_weighting']}'.")
    cfg["embeddings"]["fp16"] = parse_fp16(cfg["embeddings"]["fp16"])
    if not str(cfg["data"]["label_delimiter"]):
        raise ValueError("label_delimiter must not be empty.")
