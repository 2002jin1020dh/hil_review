"""Data schema for episodes, keyframes, VLM suggestions, human reviews and ground truth.

Design rules (see DECISIONS.md D-02, D-03):
* The labeling unit is a *keyframe* (a frame sampled at a fixed stride, plus the last frame).
* The human review unit is an *episode* (all of its keyframes are reviewed together).
* Four labels exist; only SUCCESS/FAILURE are trainable. AMBIGUOUS/INVALID are excluded and counted.
* Every record carries ``data_kind`` (real vs synthetic) so synthetic data can never be mistaken for real data.
* Timing fields carry ``timing_source`` ("measured" | "simulated" | "missing").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = "0.1.0"


class Label(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"

    @classmethod
    def parse(cls, value: "Label | str") -> "Label":
        if isinstance(value, Label):
            return value
        if not isinstance(value, str):
            raise ValueError(f"label must be a string, got {type(value).__name__}")
        v = value.strip().lower()
        for member in cls:
            if member.value == v:
                return member
        raise ValueError(f"unknown label {value!r}; expected one of {[m.value for m in cls]}")


def parse_optional_label(value: Any) -> Label | None:
    return None if value is None else Label.parse(value)


def label_to_target(label: Label | None) -> int | None:
    """Training target for the reward classifier: SUCCESS->1, FAILURE->0, anything else -> None (excluded)."""
    if label is Label.SUCCESS:
        return 1
    if label is Label.FAILURE:
        return 0
    return None


class DataKind(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


TIMING_SOURCES = ("measured", "simulated", "missing")
REVIEW_PURPOSES = ("pool_review", "audit", "audit_repeat")
VLM_STATUSES = ("ok", "parse_error", "request_error")
CONFIDENCE_SOURCES = ("logprob", "self_reported", "mock", "none")


def _req_str(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _finite_or_none(name: str, value: Any) -> None:
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number or None, got {value!r}")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    session_id: str  # collection session (day/lighting/setup); used for grouped splits
    collection_mode: str  # e.g. "success_attempt", "near_miss", "clear_failure", "policy_rollout"
    n_frames: int
    fps: float
    data_kind: DataKind
    source: str = ""  # e.g. "lerobot:<repo_id>#episode=<idx>" or "synthetic:<config_hash>"
    eligible: bool = True
    ineligible_reason: str | None = None

    def __post_init__(self) -> None:
        _req_str("episode_id", self.episode_id)
        _req_str("session_id", self.session_id)
        _req_str("collection_mode", self.collection_mode)
        if not isinstance(self.n_frames, int) or self.n_frames <= 0:
            raise ValueError(f"n_frames must be a positive int, got {self.n_frames!r}")
        if not (isinstance(self.fps, (int, float)) and self.fps > 0):
            raise ValueError(f"fps must be > 0, got {self.fps!r}")
        if not self.eligible and not self.ineligible_reason:
            raise ValueError("ineligible episodes must state ineligible_reason")
        object.__setattr__(self, "data_kind", DataKind(self.data_kind))

    @property
    def duration_s(self) -> float:
        return self.n_frames / float(self.fps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "collection_mode": self.collection_mode,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "data_kind": self.data_kind.value,
            "source": self.source,
            "eligible": self.eligible,
            "ineligible_reason": self.ineligible_reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Episode":
        return cls(
            episode_id=d["episode_id"],
            session_id=d["session_id"],
            collection_mode=d["collection_mode"],
            n_frames=int(d["n_frames"]),
            fps=float(d["fps"]),
            data_kind=DataKind(d["data_kind"]),
            source=d.get("source", ""),
            eligible=bool(d.get("eligible", True)),
            ineligible_reason=d.get("ineligible_reason"),
        )


def make_sample_id(episode_id: str, frame_index: int) -> str:
    return f"{episode_id}/f{frame_index:06d}"


@dataclass(frozen=True)
class Keyframe:
    sample_id: str
    episode_id: str
    frame_index: int
    timestamp_s: float  # relative to episode start
    frame_refs: Mapping[str, str] = field(default_factory=dict)  # camera name -> image path / URI
    features: tuple[float, ...] | None = None  # synthetic features or optional embeddings
    auto_valid: bool = True  # result of automatic validity checks (decodable, not black/frozen)

    def __post_init__(self) -> None:
        _req_str("sample_id", self.sample_id)
        _req_str("episode_id", self.episode_id)
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError(f"frame_index must be a non-negative int, got {self.frame_index!r}")
        if self.sample_id != make_sample_id(self.episode_id, self.frame_index):
            raise ValueError(
                f"sample_id {self.sample_id!r} != expected {make_sample_id(self.episode_id, self.frame_index)!r}"
            )
        _finite_or_none("timestamp_s", self.timestamp_s)
        object.__setattr__(self, "frame_refs", dict(self.frame_refs))
        if self.features is not None:
            object.__setattr__(self, "features", tuple(float(x) for x in self.features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "frame_refs": dict(self.frame_refs),
            "features": list(self.features) if self.features is not None else None,
            "auto_valid": self.auto_valid,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Keyframe":
        feats = d.get("features")
        return cls(
            sample_id=d["sample_id"],
            episode_id=d["episode_id"],
            frame_index=int(d["frame_index"]),
            timestamp_s=float(d["timestamp_s"]),
            frame_refs=dict(d.get("frame_refs") or {}),
            features=tuple(feats) if feats is not None else None,
            auto_valid=bool(d.get("auto_valid", True)),
        )


@dataclass(frozen=True)
class VLMSuggestion:
    sample_id: str
    episode_id: str
    label: Label | None  # None when the request or parsing failed
    p_success: float | None  # model score in [0, 1]; NOT a calibrated probability unless calibration is measured
    confidence_source: str
    model_id: str
    prompt_version: str
    status: str
    data_kind: DataKind
    latency_s: float | None = None
    latency_source: str = "missing"  # "measured" | "simulated" | "missing"
    raw_response: str | None = None

    def __post_init__(self) -> None:
        _req_str("sample_id", self.sample_id)
        _req_str("episode_id", self.episode_id)
        if self.status not in VLM_STATUSES:
            raise ValueError(f"status must be one of {VLM_STATUSES}, got {self.status!r}")
        if self.confidence_source not in CONFIDENCE_SOURCES:
            raise ValueError(f"confidence_source must be one of {CONFIDENCE_SOURCES}, got {self.confidence_source!r}")
        if self.latency_source not in TIMING_SOURCES:
            raise ValueError(f"latency_source must be one of {TIMING_SOURCES}")
        object.__setattr__(self, "label", parse_optional_label(self.label))
        object.__setattr__(self, "data_kind", DataKind(self.data_kind))
        if self.status != "ok" and self.label is not None:
            raise ValueError("failed VLM suggestions must have label=None")
        if self.status == "ok" and self.label is None:
            raise ValueError("ok VLM suggestions must have a label")
        _finite_or_none("p_success", self.p_success)
        if self.p_success is not None and not (0.0 <= self.p_success <= 1.0):
            raise ValueError(f"p_success must be in [0, 1], got {self.p_success}")
        _finite_or_none("latency_s", self.latency_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "label": self.label.value if self.label is not None else None,
            "p_success": self.p_success,
            "confidence_source": self.confidence_source,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "status": self.status,
            "data_kind": self.data_kind.value,
            "latency_s": self.latency_s,
            "latency_source": self.latency_source,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VLMSuggestion":
        return cls(
            sample_id=d["sample_id"],
            episode_id=d["episode_id"],
            label=parse_optional_label(d.get("label")),
            p_success=d.get("p_success"),
            confidence_source=d["confidence_source"],
            model_id=d["model_id"],
            prompt_version=d["prompt_version"],
            status=d["status"],
            data_kind=DataKind(d["data_kind"]),
            latency_s=d.get("latency_s"),
            latency_source=d.get("latency_source", "missing"),
            raw_response=d.get("raw_response"),
        )


@dataclass(frozen=True)
class ReviewRecord:
    """One human (or simulated human) review of one whole episode."""

    episode_id: str
    reviewer_id: str
    purpose: str  # "pool_review" (assisted, VLM shown) | "audit" (blind) | "audit_repeat" (blind, 2nd pass)
    shown_vlm: bool
    labels: Mapping[str, Label]  # sample_id -> label, for every keyframe of the episode
    active_time_s: float | None
    timing_source: str
    data_kind: DataKind
    started_at: str | None = None
    ended_at: str | None = None
    review_order: int | None = None  # position in the review session (to study fatigue/order effects)
    session_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        _req_str("episode_id", self.episode_id)
        _req_str("reviewer_id", self.reviewer_id)
        if self.purpose not in REVIEW_PURPOSES:
            raise ValueError(f"purpose must be one of {REVIEW_PURPOSES}, got {self.purpose!r}")
        if self.purpose in ("audit", "audit_repeat") and self.shown_vlm:
            raise ValueError("audit reviews must be blind (shown_vlm=False)")
        if self.timing_source not in TIMING_SOURCES:
            raise ValueError(f"timing_source must be one of {TIMING_SOURCES}")
        if self.timing_source == "missing" and self.active_time_s is not None:
            raise ValueError("timing_source='missing' requires active_time_s=None")
        if self.timing_source != "missing" and self.active_time_s is None:
            raise ValueError("active_time_s required unless timing_source='missing'")
        _finite_or_none("active_time_s", self.active_time_s)
        if self.active_time_s is not None and self.active_time_s < 0:
            raise ValueError("active_time_s must be >= 0")
        if not self.labels:
            raise ValueError("a review record must contain at least one keyframe label")
        object.__setattr__(self, "labels", {k: Label.parse(v) for k, v in self.labels.items()})
        object.__setattr__(self, "data_kind", DataKind(self.data_kind))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "reviewer_id": self.reviewer_id,
            "purpose": self.purpose,
            "shown_vlm": self.shown_vlm,
            "labels": {k: v.value for k, v in sorted(self.labels.items())},
            "active_time_s": self.active_time_s,
            "timing_source": self.timing_source,
            "data_kind": self.data_kind.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "review_order": self.review_order,
            "session_id": self.session_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReviewRecord":
        return cls(
            episode_id=d["episode_id"],
            reviewer_id=d["reviewer_id"],
            purpose=d["purpose"],
            shown_vlm=bool(d["shown_vlm"]),
            labels={k: Label.parse(v) for k, v in d["labels"].items()},
            active_time_s=d.get("active_time_s"),
            timing_source=d["timing_source"],
            data_kind=DataKind(d["data_kind"]),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            review_order=d.get("review_order"),
            session_id=d.get("session_id", ""),
            notes=d.get("notes", ""),
        )


# Ground-truth keyframe states for termination replay and synthetic evaluation.
STATE_SUCCESS = "S"
STATE_FAILURE = "F"
STATE_UNKNOWN = "U"  # ambiguous / invalid / unlabeled


def label_to_state(label: Label | None) -> str:
    if label is Label.SUCCESS:
        return STATE_SUCCESS
    if label is Label.FAILURE:
        return STATE_FAILURE
    return STATE_UNKNOWN


@dataclass(frozen=True)
class SyntheticKeyframeTruth:
    """Ground truth that exists ONLY for synthetic data (real data has no oracle)."""

    sample_id: str
    state: str  # "S" or "F"
    boundary: bool  # within one keyframe of a state change (hard for annotators and VLMs)
    hover: bool  # object held above/near the container without being inside ("fake success" bait)
    occluded: bool  # visually ambiguous (e.g. gripper hides the object)

    def __post_init__(self) -> None:
        if self.state not in (STATE_SUCCESS, STATE_FAILURE):
            raise ValueError("synthetic truth state must be 'S' or 'F'")
