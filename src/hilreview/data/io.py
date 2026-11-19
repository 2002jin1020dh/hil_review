"""JSONL reading/writing for schema records. Append-friendly so interactive sessions are crash-safe."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Write records (objects with ``to_dict`` or plain dicts) atomically. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: str | Path, record: Any) -> None:
    """Append one record and flush+fsync, so a crash loses at most the record being written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = record.to_dict() if hasattr(record, "to_dict") else record
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def read_jsonl(path: str | Path, parser: Callable[[dict[str, Any]], T] | None = None) -> list[T] | list[dict]:
    rows = list(iter_jsonl(path))
    return [parser(r) for r in rows] if parser is not None else rows
