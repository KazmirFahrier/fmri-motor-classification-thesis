from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL_KEYS = {
    "task",
    "classes",
    "input",
    "loader",
    "model",
    "optimizer",
    "scheduler",
    "training",
    "evaluation",
    "seed",
    "paths",
}

REQUIRED_NESTED_KEYS = {
    "classes": {"names"},
    "input": {"target_shape", "normalization", "hrf_shift"},
    "loader": {"batch_size", "num_workers", "clip_lengths", "clip_length", "clip_stride", "clip_window_stride"},
    "model": {"name", "in_channels", "num_classes", "base_channels", "dropout"},
    "optimizer": {"name", "lr", "weight_decay"},
    "scheduler": {"name", "warmup_epochs", "min_lr"},
    "training": {
        "epochs",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "monitor",
        "monitor_mode",
        "amp",
        "max_grad_norm",
        "label_smoothing",
        "deterministic",
    },
    "evaluation": {"bootstrap_samples", "ci_alpha", "metrics"},
    "paths": {"artifacts_root", "cache_root"},
}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: Dict[str, Any]


def load_config(path: str | Path) -> LoadedConfig:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top-level: {cfg_path}")
    validate_config(data)
    return LoadedConfig(path=cfg_path, data=data)


def validate_config(cfg: Dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(cfg.keys())
    extra = set(cfg.keys()).difference(REQUIRED_TOP_LEVEL_KEYS)
    if missing:
        raise ValueError(f"Missing required top-level config keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected top-level config keys (strict mode): {sorted(extra)}")

    for section, required_keys in REQUIRED_NESTED_KEYS.items():
        value = cfg.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"Section '{section}' must be a mapping")
        section_missing = required_keys.difference(value.keys())
        if section_missing:
            raise ValueError(
                f"Missing keys in section '{section}': {sorted(section_missing)}"
            )

    class_names = cfg["classes"]["names"]
    if not isinstance(class_names, list) or not class_names or not all(
        isinstance(c, str) for c in class_names
    ):
        raise ValueError("classes.names must be a non-empty list[str]")

    if int(cfg["model"]["num_classes"]) != len(class_names):
        raise ValueError(
            "model.num_classes must match len(classes.names)"
        )

    task = str(cfg["task"]).lower()
    if task not in {"volume", "clip"}:
        raise ValueError("task must be one of: volume, clip")

    target_shape = cfg["input"]["target_shape"]
    if (
        not isinstance(target_shape, list)
        or len(target_shape) != 3
        or not all(isinstance(v, int) and v > 0 for v in target_shape)
    ):
        raise ValueError("input.target_shape must be [D, H, W] positive ints")

    clip_lengths = cfg["loader"]["clip_lengths"]
    if (
        not isinstance(clip_lengths, list)
        or not clip_lengths
        or not all(isinstance(v, int) and v > 0 for v in clip_lengths)
    ):
        raise ValueError("loader.clip_lengths must be non-empty list[int]")

    if int(cfg["loader"]["clip_length"]) not in set(int(v) for v in clip_lengths):
        raise ValueError("loader.clip_length must be present in loader.clip_lengths")

    if cfg["training"]["monitor"] not in cfg["evaluation"]["metrics"]:
        raise ValueError(
            "training.monitor must be listed in evaluation.metrics"
        )

    if cfg["training"]["monitor_mode"] not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be 'min' or 'max'")


def dump_config(cfg: Dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
