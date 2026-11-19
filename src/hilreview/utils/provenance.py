"""Provenance: who/what/when produced a result. Attached to every results directory."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hilreview import __version__


def utc_now_iso() -> str:
    """Current UTC time, ISO-8601 with seconds precision and 'Z' suffix."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding (sorted keys, no whitespace variation)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def config_hash(cfg: Any) -> str:
    """Short SHA-256 of the canonical JSON form of a config."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Commit hash and dirty flag of the code repository, or a note that git info is unavailable.

    'dirty' means tracked changes or untracked files OUTSIDE results/ (outputs of runs must not mark the code dirty).
    """
    cwd = str(repo_dir) if repo_dir is not None else str(Path(__file__).resolve().parents[3])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results"], cwd=cwd, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception as exc:  # no git, not a repo, or no commits yet
        return {"commit": None, "dirty": None, "note": f"git info unavailable: {type(exc).__name__}"}


def environment_info() -> dict[str, Any]:
    return {
        "hilreview_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
