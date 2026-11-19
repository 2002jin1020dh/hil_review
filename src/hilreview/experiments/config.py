"""Experiment configuration loading and validation (YAML)."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from hilreview.labeling.aggregate import ConditionSpec

DEFAULTS: dict[str, Any] = {
    "splits": {"audit_fraction": 0.25, "val_fraction_of_pool": 0.2, "group_by": "episode_id",
               "stratify_by": ["session_id", "collection_mode"], "seed": 20261017},
    "replicates": {"n": 5, "base_seed": 1000},
    "review_order_seed": 77,
    "audit_repeat_fraction": 0.3,
    "classifier": {"l2": 1e-3, "lr": 0.1, "epochs": 30, "batch_size": 64},
    "threshold": {"policy": "fixed", "value": 0.5},
    "secondary_threshold": {"policy": "val_fbeta", "beta": 0.5},
    "termination": {"debounce_k": 1, "tolerance": 0},
    "effort": {"idle_cap_s": 120.0},
    "bootstrap": {"n_boot": 1000, "seed": 4242},
    "output_root": "results",
    "report_lang": "ko",  # report wording: "ko" (default, user request) or "en"; numbers are identical
}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    cfg = _deep_merge(DEFAULTS, raw)
    name = cfg.get("experiment_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ConfigError("experiment_name must match [A-Za-z0-9_.-]+")
    data = cfg.get("data") or {}
    kind = data.get("kind")
    if kind not in ("synthetic", "files"):
        raise ConfigError("data.kind must be 'synthetic' (generated in memory) or 'files' (JSONL records; "
                          "real vs synthetic is then read from each record's data_kind)")
    if kind == "synthetic":
        cfg.setdefault("vlm", {}).setdefault("mock", {})
        cfg.setdefault("reviewer", {}).setdefault("simulated", {})
        data.setdefault("synthetic", {})
    else:
        files = data.get("files") or {}
        for key in ("episodes", "keyframes", "vlm_suggestions", "pool_reviews", "audit_reviews", "split_file"):
            if not files.get(key):
                raise ConfigError(f"data.files.{key} is required when data.kind == 'files'")
    conds = cfg.get("conditions")
    if not isinstance(conds, list) or not conds:
        raise ConfigError("conditions must be a non-empty list")
    names = [c.get("name") for c in conds]
    if len(set(names)) != len(names):
        raise ConfigError("condition names must be unique")
    for c in conds:
        try:
            ConditionSpec(name=c["name"], policy=c["policy"], fraction=float(c["fraction"]),
                          budget_unit=c.get("budget_unit", "episodes"))
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"invalid condition {c!r}: {exc}") from exc
    if int(cfg["replicates"]["n"]) < 1:
        raise ConfigError("replicates.n must be >= 1")
    if cfg["threshold"]["policy"] not in ("fixed", "val_fbeta"):
        raise ConfigError("threshold.policy must be 'fixed' or 'val_fbeta'")
    if int(cfg["termination"]["debounce_k"]) < 1 or int(cfg["termination"]["tolerance"]) < 0:
        raise ConfigError("termination.debounce_k >= 1 and tolerance >= 0 required")
    if cfg.get("report_lang") not in ("ko", "en"):
        raise ConfigError("report_lang must be 'ko' or 'en'")
    return cfg


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = validate_config(raw)
    cfg["_config_path"] = str(path)
    return cfg
