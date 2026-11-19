"""VLM label suggestions: prompt, response parsing, a mock labeler, and a local HTTP client.

* No cloud VLMs (project rule). The HTTP client targets a LOCAL OpenAI-compatible server
  (e.g. vLLM or llama.cpp server running Qwen3-VL on the user's GPU laptop). It uses only the
  standard library. It is unit-tested against a stub server here but has NOT been run against a real model.
* p_success comes from answer-token log-probabilities when the server returns them
  (confidence_source="logprob"); otherwise it is None (confidence_source="none").
  p_success is a ranking score, NOT a calibrated probability, unless calibration is measured
  on the labeling pool (never on the audit set).
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from hilreview.data.schema import (
    STATE_SUCCESS,
    DataKind,
    Keyframe,
    Label,
    SyntheticKeyframeTruth,
    VLMSuggestion,
)
from hilreview.utils.rng import make_rng

PROMPT_VERSION = "succ-yesno-v1"

SYSTEM_PROMPT = (
    "You are a careful annotator of robot manipulation camera images. "
    "You judge only what is visible in the images."
)

USER_PROMPT_TEMPLATE = (
    "Task: {task}\n"
    "The images show the same moment from {n_views} camera view(s): {views}.\n"
    "Success means: {success_definition}\n"
    "Question: at this exact moment, is the task successfully completed?\n"
    "Answer with exactly one word: yes, no, or unsure. "
    "Answer unsure only if the object or the container is not visible enough to decide."
)

DEFAULT_SUCCESS_DEFINITION = (
    "the target object is entirely inside the container, the gripper is open and no longer holding it, "
    "and the object is at rest. An object on the rim, held above the container, or still gripped is NOT success."
)

_YES = {"yes", "y", "true"}
_NO = {"no", "n", "false"}
_UNSURE = {"unsure", "unclear", "uncertain", "unknown"}


def build_user_prompt(task: str, views: Sequence[str], success_definition: str = DEFAULT_SUCCESS_DEFINITION) -> str:
    return USER_PROMPT_TEMPLATE.format(
        task=task, n_views=len(views), views=", ".join(views) or "unknown", success_definition=success_definition
    )


def parse_answer_text(text: str | None) -> Label | None:
    """Map the first word of the answer to a label; None if unparseable."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    m = re.search(r"[A-Za-z]+", cleaned)
    if not m:
        return None
    w = m.group(0).lower()
    if w in _YES:
        return Label.SUCCESS
    if w in _NO:
        return Label.FAILURE
    if w in _UNSURE:
        return Label.AMBIGUOUS
    return None


def p_success_from_logprobs(content_logprobs: Sequence[Mapping[str, Any]] | None, max_positions: int = 3) -> float | None:
    """P(yes) / (P(yes) + P(no)) at the first non-whitespace answer position.

    ``content_logprobs`` is OpenAI-style ``choices[0].logprobs.content``: a list of positions, each with
    ``token``, ``logprob`` and ``top_logprobs`` ([{token, logprob}, ...]). Returns None if neither a yes- nor
    a no-like token appears among the top candidates.
    """
    if not content_logprobs:
        return None
    for pos in list(content_logprobs)[:max_positions]:
        tok = str(pos.get("token", ""))
        if not tok.strip():
            continue  # skip leading whitespace/newline tokens
        cands = list(pos.get("top_logprobs") or [])
        if not cands and "logprob" in pos:
            cands = [{"token": tok, "logprob": pos["logprob"]}]
        p_yes = p_no = 0.0
        for c in cands:
            t = str(c.get("token", "")).strip().lower()
            lp = c.get("logprob")
            if lp is None or not math.isfinite(float(lp)):
                continue
            if t in _YES:
                p_yes += math.exp(float(lp))
            elif t in _NO:
                p_no += math.exp(float(lp))
        if p_yes + p_no == 0.0:
            return None
        return p_yes / (p_yes + p_no)
    return None


class VLMLabeler(Protocol):
    model_id: str
    prompt_version: str

    def suggest(self, keyframes: Sequence[Keyframe]) -> list[VLMSuggestion]: ...


# --------------------------------------------------------------------------------------------
# Mock labeler (SYNTHETIC ONLY)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MockVLMProfile:
    """Error profile of the mock VLM. These numbers are invented placeholders, NOT estimates of Qwen3-VL."""

    base_error: float = 0.05  # flip probability on clear keyframes
    boundary_error: float = 0.30  # flip probability within one keyframe of a state change
    occluded_error: float = 0.35
    hover_fake_success: float = 0.45  # P(says success | object hovering over container, truly failure)
    unsure_rate: float = 0.02  # answers "unsure"
    request_error_rate: float = 0.01  # simulated server/parse failures
    correct_margin_beta: tuple[float, float] = (4.0, 1.5)  # margin |2p-1| when the answer is right
    wrong_margin_beta: tuple[float, float] = (1.5, 3.0)  # margin when wrong (informative but imperfect)
    latency_s: float = 0.8  # simulated seconds per call


class MockVLMLabeler:
    """Synthetic VLM driven by synthetic ground truth. Every output has data_kind='synthetic'."""

    prompt_version = "mock"

    def __init__(self, truth: Mapping[str, SyntheticKeyframeTruth], profile: MockVLMProfile, seed: int):
        self.truth = truth
        self.profile = profile
        self.seed = seed
        self.model_id = "mock-vlm"

    def suggest(self, keyframes: Sequence[Keyframe]) -> list[VLMSuggestion]:
        out = []
        p = self.profile
        for kf in keyframes:
            rng = make_rng(self.seed, "mock_vlm", kf.sample_id)  # per-sample stream: order-independent
            t = self.truth[kf.sample_id]
            if rng.random() < p.request_error_rate:
                out.append(VLMSuggestion(kf.sample_id, kf.episode_id, None, None, "none", self.model_id,
                                         self.prompt_version, "request_error", DataKind.SYNTHETIC,
                                         p.latency_s, "simulated"))
                continue
            true_succ = t.state == STATE_SUCCESS
            if t.hover and not true_succ:
                err = max(p.hover_fake_success, p.base_error)
            elif t.occluded:
                err = p.occluded_error
            elif t.boundary:
                err = p.boundary_error
            else:
                err = p.base_error
            wrong = rng.random() < err
            says_succ = (not true_succ) if wrong else true_succ
            a, b = p.wrong_margin_beta if wrong else p.correct_margin_beta
            margin = float(rng.beta(a, b))
            p_succ = 0.5 + margin / 2 if says_succ else 0.5 - margin / 2
            label = Label.SUCCESS if says_succ else Label.FAILURE
            if rng.random() < p.unsure_rate:
                label = Label.AMBIGUOUS
            out.append(VLMSuggestion(kf.sample_id, kf.episode_id, label, float(np.clip(p_succ, 0.0, 1.0)), "mock",
                                     self.model_id, self.prompt_version, "ok", DataKind.SYNTHETIC,
                                     p.latency_s, "simulated"))
        return out


# --------------------------------------------------------------------------------------------
# Local OpenAI-compatible HTTP client (REAL data; unverified against a real model in this sandbox)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAICompatConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    api_key_env: str | None = None  # NAME of an env var holding a key; the value is never logged or stored
    timeout_s: float = 60.0
    max_tokens: int = 3
    temperature: float = 0.0
    request_logprobs: bool = True
    top_logprobs: int = 5
    camera_order: tuple[str, ...] = field(default_factory=tuple)  # empty = sorted camera names


def _data_url(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def is_local_or_private_url(url: str) -> bool:
    """True for loopback, RFC1918/link-local/ULA addresses, 'localhost' and '*.local' (mDNS) hosts.

    Project rule: robot images are private and no cloud VLMs are used, so images may only be sent to the
    user's own machines unless they explicitly opt in (allow_remote_host=True).
    """
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


class OpenAICompatVLMLabeler:
    def __init__(
        self,
        cfg: OpenAICompatConfig,
        task: str,
        success_definition: str = DEFAULT_SUCCESS_DEFINITION,
        data_kind: DataKind = DataKind.REAL,
        allow_remote_host: bool = False,
    ):
        if not allow_remote_host and not is_local_or_private_url(cfg.base_url):
            raise ValueError(
                f"refusing to send robot images to non-local host in {cfg.base_url!r}. Use a VLM server on your own "
                "machine/LAN, or pass allow_remote_host=True (--allow-remote-host) after deciding this deliberately."
            )
        self.cfg = cfg
        self.task = task
        self.success_definition = success_definition
        self.data_kind = DataKind(data_kind)
        self.model_id = cfg.model
        self.prompt_version = PROMPT_VERSION

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if key:
                h["Authorization"] = f"Bearer {key}"
        return h

    def build_request(self, kf: Keyframe) -> dict[str, Any]:
        cams = list(self.cfg.camera_order) or sorted(kf.frame_refs)
        missing = [c for c in cams if c not in kf.frame_refs]
        if missing:
            raise FileNotFoundError(f"{kf.sample_id}: no frame for camera(s) {missing}")
        content: list[dict[str, Any]] = [{"type": "text", "text": build_user_prompt(self.task, cams,
                                                                                     self.success_definition)}]
        for c in cams:
            content.append({"type": "image_url", "image_url": {"url": _data_url(kf.frame_refs[c])}})
        req: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.request_logprobs:
            req["logprobs"] = True
            req["top_logprobs"] = self.cfg.top_logprobs
        return req

    def suggest_one(self, kf: Keyframe) -> VLMSuggestion:
        t0 = time.perf_counter()
        raw: str | None = None
        try:
            body = json.dumps(self.build_request(kf)).encode("utf-8")
            req = urllib.request.Request(self.cfg.base_url.rstrip("/") + "/chat/completions", data=body,
                                         headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return VLMSuggestion(kf.sample_id, kf.episode_id, None, None, "none", self.model_id, self.prompt_version,
                                 "request_error", self.data_kind, time.perf_counter() - t0, "measured",
                                 f"{type(exc).__name__}: {exc}"[:500])
        latency = time.perf_counter() - t0
        try:
            choice = payload["choices"][0]
            raw = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return VLMSuggestion(kf.sample_id, kf.episode_id, None, None, "none", self.model_id, self.prompt_version,
                                 "parse_error", self.data_kind, latency, "measured", json.dumps(payload)[:500])
        label = parse_answer_text(raw)
        if label is None:
            return VLMSuggestion(kf.sample_id, kf.episode_id, None, None, "none", self.model_id, self.prompt_version,
                                 "parse_error", self.data_kind, latency, "measured", (raw or "")[:500])
        lp = (choice.get("logprobs") or {}).get("content") if isinstance(choice.get("logprobs"), dict) else None
        p = p_success_from_logprobs(lp)
        return VLMSuggestion(kf.sample_id, kf.episode_id, label, p, "logprob" if p is not None else "none",
                             self.model_id, self.prompt_version, "ok", self.data_kind, latency, "measured",
                             (raw or "")[:500])

    def suggest(self, keyframes: Sequence[Keyframe]) -> list[VLMSuggestion]:
        return [self.suggest_one(kf) for kf in keyframes]
