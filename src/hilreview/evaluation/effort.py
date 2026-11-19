"""Human review effort accounting (DECISIONS.md D-07).

Times come from ReviewRecord.active_time_s; their provenance (measured/simulated/missing) is carried
through. Long idle periods inflate measured times, so totals are reported both raw and capped at
``idle_cap_s`` (with the number of capped episodes); neither is hidden.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

from hilreview.data.schema import Keyframe, ReviewRecord, VLMSuggestion


def summarize_review_effort(
    selected_episodes: Sequence[str],
    pool_reviews: Mapping[str, ReviewRecord],
    keyframes_by_episode: Mapping[str, Sequence[Keyframe]],
    vlm_by_sample: Mapping[str, VLMSuggestion],
    idle_cap_s: float = 120.0,
) -> dict:
    if not selected_episodes:
        return {
            "n_episodes_reviewed": 0, "n_keyframes_reviewed": 0, "n_corrections": 0, "correction_rate": None,
            "human_time_s": 0.0, "human_time_s_capped": 0.0, "n_time_capped": 0, "n_time_missing": 0,
            "mean_s_per_episode": None, "median_s_per_episode": None, "s_per_keyframe": None,
            "timing_source": "not_applicable",
        }
    times, sources = [], set()
    n_kf = n_corr = n_missing = 0
    for eid in selected_episodes:
        rec = pool_reviews[eid]
        sources.add(rec.timing_source)
        if rec.active_time_s is None:
            n_missing += 1
        else:
            times.append(float(rec.active_time_s))
        for kf in keyframes_by_episode[eid]:
            if not kf.auto_valid:
                continue
            n_kf += 1
            s = vlm_by_sample.get(kf.sample_id)
            if rec.labels.get(kf.sample_id) is not (None if s is None else s.label):
                n_corr += 1
    src = sources.pop() if len(sources) == 1 else "mixed"
    capped = [min(t, idle_cap_s) for t in times]
    # math.fsum is exactly rounded on every Python version (built-in sum() changed in 3.12), so totals are
    # bit-identical across interpreters
    total = math.fsum(times) if times else None
    return {
        "n_episodes_reviewed": len(selected_episodes),
        "n_keyframes_reviewed": n_kf,
        "n_corrections": n_corr,  # keyframes whose final human label differs from the VLM suggestion
        "correction_rate": (n_corr / n_kf) if n_kf else None,
        # totals over episodes WITH timing; n_time_missing says how many lacked it
        "human_time_s": total,
        "human_time_s_capped": math.fsum(capped) if times else None,
        "n_time_capped": sum(1 for t in times if t > idle_cap_s),
        "n_time_missing": n_missing,
        "mean_s_per_episode": statistics.fmean(times) if times else None,
        "median_s_per_episode": statistics.median(times) if times else None,
        "s_per_keyframe": (total / n_kf) if (times and n_kf and n_missing == 0) else None,
        "timing_source": src,
    }


def summarize_vlm_cost(sample_ids: Sequence[str], vlm_by_sample: Mapping[str, VLMSuggestion]) -> dict:
    n_calls = n_ok = n_err = n_unsure = 0
    lat_values: list[float] = []
    lat_sources: set[str] = set()
    for sid in sample_ids:
        s = vlm_by_sample.get(sid)
        if s is None:
            continue
        n_calls += 1
        if s.status == "ok":
            n_ok += 1
            if s.label is not None and s.label.value == "ambiguous":
                n_unsure += 1
        else:
            n_err += 1
        if s.latency_s is not None:
            lat_values.append(s.latency_s)
            lat_sources.add(s.latency_source)
    return {
        "n_calls": n_calls,
        "n_ok": n_ok,
        "n_errors": n_err,
        "n_unsure": n_unsure,
        "total_latency_s": math.fsum(lat_values) if lat_values else None,
        "latency_source": (lat_sources.pop() if len(lat_sources) == 1 else ("mixed" if lat_sources else "missing")),
    }
