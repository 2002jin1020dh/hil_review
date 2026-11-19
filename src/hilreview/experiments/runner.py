"""Experiment runner: conditions x replicates on fixed data, with audit-only final evaluation.

Pipeline (identical for synthetic and real data up to classifier scoring):
  1. data: episodes, keyframes, ONE set of VLM suggestions, ONE full assisted pool review,
     blind audit labels (+ optional blind repeat pass), and a FROZEN split.
  2. per replicate r and condition c:
       select reviewed episodes (policy, budget, seeded) -> final labels -> effort, label quality
       (synthetic) train stand-in classifier on train labels -> threshold on VAL with the condition's labels
       -> score audit keyframes -> keyframe metrics + episode termination replay vs BLIND audit labels.
  3. aggregate over replicates; write JSON/CSV/report with provenance.

Common random numbers: within replicate r, every condition uses the same selection RNG seed and the same
classifier-training seed, so between-condition differences are not inflated by unrelated randomness.
Audit isolation: audit labels are read ONLY inside ``evaluate_on_audit``. Row fingerprints
(selected/labels/model/threshold) let tests prove that perturbing audit labels changes nothing upstream.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from hilreview.data.io import read_jsonl
from hilreview.data.schema import (
    STATE_UNKNOWN,
    DataKind,
    Episode,
    Keyframe,
    Label,
    ReviewRecord,
    SyntheticKeyframeTruth,
    VLMSuggestion,
    label_to_state,
    label_to_target,
)
from hilreview.data.splits import SplitSpec, assign_splits, check_split_integrity
from hilreview.data.synthetic import SyntheticConfig, generate_synthetic
from hilreview.evaluation.effort import summarize_review_effort, summarize_vlm_cost
from hilreview.evaluation.metrics import (
    BinaryCounts,
    binary_rates,
    cluster_bootstrap_ci,
    cohen_kappa,
    count_binary,
    paired_cluster_bootstrap_diff,
    summarize_values,
)
from hilreview.evaluation.termination import fst_indicator, replay_all, summarize_replays
from hilreview.labeling.aggregate import ConditionLabels, ConditionSpec, build_condition_labels
from hilreview.labeling.vlm import MockVLMLabeler, MockVLMProfile
from hilreview.models.logistic import LogisticRegressionSGD
from hilreview.models.threshold import select_threshold
from hilreview.review.simulated import SimulatedReviewer, SimulatedReviewerProfile
from hilreview.utils.numeric import round_half_up
from hilreview.utils.provenance import config_hash, environment_info, git_info, utc_now_iso
from hilreview.utils.rng import derive_seed, make_rng


@dataclass
class ExperimentData:
    data_kind: DataKind
    episodes: list[Episode]
    keyframes_by_episode: dict[str, list[Keyframe]]
    vlm_by_sample: dict[str, VLMSuggestion]
    pool_reviews: dict[str, ReviewRecord]
    audit_reviews: dict[str, ReviewRecord]
    audit_repeat: dict[str, ReviewRecord]
    split: dict[str, str]
    truth: dict[str, SyntheticKeyframeTruth] | None = None

    def ids(self, *split_names: str) -> list[str]:
        return sorted(eid for eid, s in self.split.items() if s in split_names)


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------------


def _synthetic_config(d: Mapping[str, Any]) -> SyntheticConfig:
    d = dict(d)
    for key in ("duration_s", "onset_frac"):
        if key in d:
            d[key] = tuple(d[key])
    return SyntheticConfig(**d)


def prepare_synthetic(cfg: Mapping[str, Any]) -> ExperimentData:
    scfg = _synthetic_config(cfg["data"]["synthetic"])
    ds = generate_synthetic(scfg)
    spec = SplitSpec(**{**cfg["splits"], "stratify_by": tuple(cfg["splits"]["stratify_by"])})
    split = assign_splits(ds.episodes, spec)
    check_split_integrity(ds.episodes, split, ds.keyframes_by_episode, spec.group_by)
    eligible = [e.episode_id for e in ds.episodes if e.eligible]
    kfs = [kf for eid in eligible for kf in ds.keyframes_by_episode[eid] if kf.auto_valid]
    vlm = MockVLMLabeler(ds.truth, MockVLMProfile(**_tuplify(cfg["vlm"]["mock"])), derive_seed(scfg.seed, "vlm"))
    vlm_by = {s.sample_id: s for s in vlm.suggest(kfs)}
    reviewer = SimulatedReviewer(ds.truth, SimulatedReviewerProfile(**cfg["reviewer"]["simulated"]),
                                 derive_seed(scfg.seed, "reviewer"))
    pool = sorted(eid for eid, s in split.items() if s in ("train", "val"))
    audit = sorted(eid for eid, s in split.items() if s == "audit")
    order = [pool[i] for i in make_rng(cfg["review_order_seed"], "pool_order").permutation(len(pool))]
    pool_reviews = reviewer.review_pool(order, ds.keyframes_by_episode, vlm_by)
    a_order = [audit[i] for i in make_rng(cfg["review_order_seed"], "audit_order").permutation(len(audit))]
    audit_reviews = reviewer.label_blind(a_order, ds.keyframes_by_episode, "audit")
    n_rep = round_half_up(float(cfg["audit_repeat_fraction"]) * len(audit))
    rep_ids = [audit[i] for i in make_rng(cfg["review_order_seed"], "audit_repeat").permutation(len(audit))[:n_rep]]
    audit_repeat = reviewer.label_blind(rep_ids, ds.keyframes_by_episode, "audit_repeat")
    return ExperimentData(DataKind.SYNTHETIC, ds.episodes, ds.keyframes_by_episode, vlm_by, pool_reviews,
                          audit_reviews, audit_repeat, split, ds.truth)


def _tuplify(d: Mapping[str, Any]) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


def _records_by_episode(path: str | Path, purpose: str) -> dict[str, ReviewRecord]:
    out: dict[str, ReviewRecord] = {}
    for r in read_jsonl(path, ReviewRecord.from_dict):
        if r.purpose != purpose:
            raise ValueError(f"{path}: expected purpose {purpose!r}, found {r.purpose!r} for {r.episode_id}")
        if r.episode_id in out:
            raise ValueError(f"{path}: duplicate review for episode {r.episode_id!r}")
        out[r.episode_id] = r
    return out


def load_files(cfg: Mapping[str, Any]) -> ExperimentData:
    """Load JSONL records. Real vs synthetic is taken from the records themselves (mixing is an error)."""
    p = cfg["data"]["files"]
    episodes = read_jsonl(p["episodes"], Episode.from_dict)
    kinds = {e.data_kind for e in episodes}
    if len(kinds) != 1:
        raise ValueError(f"episodes mix data kinds {sorted(k.value for k in kinds)}; refusing to combine them")
    kind = kinds.pop()
    kfs: dict[str, list[Keyframe]] = defaultdict(list)
    for kf in read_jsonl(p["keyframes"], Keyframe.from_dict):
        kfs[kf.episode_id].append(kf)
    for eid in kfs:
        kfs[eid].sort(key=lambda k: k.frame_index)
    vlm = {s.sample_id: s for s in read_jsonl(p["vlm_suggestions"], VLMSuggestion.from_dict)}
    with open(p["split_file"], "r", encoding="utf-8") as f:
        split = json.load(f)["episode_split"]  # FROZEN split written by `hilreview split`
    check_split_integrity(episodes, split, kfs, cfg["splits"]["group_by"])
    audit_repeat = _records_by_episode(p["audit_repeat"], "audit_repeat") if p.get("audit_repeat") else {}
    data = ExperimentData(kind, episodes, dict(kfs), vlm, _records_by_episode(p["pool_reviews"], "pool_review"),
                          _records_by_episode(p["audit_reviews"], "audit"), audit_repeat, split, None)
    other = [r for r in (*data.pool_reviews.values(), *data.audit_reviews.values(), *data.audit_repeat.values(),
                         *data.vlm_by_sample.values()) if r.data_kind is not kind]
    if other:
        raise ValueError(f"{len(other)} review/VLM records have a data_kind different from the episodes ({kind.value})")
    check_review_placement(data)
    return data


def check_review_placement(data: ExperimentData) -> None:
    """Protocol checks for real review files (DECISIONS.md D-07, D-08).

    * every eligible pool episode has exactly one assisted pool review covering all its keyframes;
    * NO audit episode appears in the assisted pool review (its blind labels would be anchored by the VLM);
    * every audit episode has a blind audit record covering all its keyframes; repeats are audit episodes only.
    """
    pool, audit = set(data.ids("train", "val")), set(data.ids("audit"))
    leaked = sorted(set(data.pool_reviews) & audit)
    if leaked:
        raise ValueError(f"audit episodes were reviewed WITH VLM suggestions (pool_review): {leaked[:5]}; "
                         "their audit labels are no longer independent")
    stray = sorted(set(data.pool_reviews) - pool)
    if stray:
        raise ValueError(f"pool reviews for episodes outside train/val: {stray[:5]}")
    for name, recs, expected in (("pool_review", data.pool_reviews, pool), ("audit", data.audit_reviews, audit)):
        missing = sorted(expected - set(recs))
        if missing:
            raise ValueError(f"{len(missing)} episode(s) lack a {name} record (e.g. {missing[:5]}); "
                             "finish the review session first")
        for eid in expected:
            need = {kf.sample_id for kf in data.keyframes_by_episode[eid]}
            if not need <= set(recs[eid].labels):
                raise ValueError(f"{name} record of {eid!r} does not cover all keyframes")
    if not set(data.audit_repeat) <= audit:
        raise ValueError("audit_repeat records must be for audit episodes only")


def prepare_data(cfg: Mapping[str, Any]) -> ExperimentData:
    return prepare_synthetic(cfg) if cfg["data"]["kind"] == "synthetic" else load_files(cfg)


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def _valid_samples(data: ExperimentData, episode_ids: Sequence[str]) -> list[str]:
    return [kf.sample_id for eid in episode_ids for kf in data.keyframes_by_episode[eid] if kf.auto_valid]


def _label_quality(labels: Mapping[str, Label | None], reference: Mapping[str, Label | None],
                   sample_ids: Sequence[str]) -> dict:
    r = count_binary([label_to_target(reference.get(s)) for s in sample_ids],
                     [label_to_target(labels.get(s)) for s in sample_ids])
    out = binary_rates(r.counts)
    out.update({"counts": r.counts.to_dict(), "n_ref_excluded": r.n_ref_missing, "n_label_excluded": r.n_pred_missing})
    return out


def _truth_labels(data: ExperimentData, sample_ids: Sequence[str]) -> dict[str, Label]:
    assert data.truth is not None
    return {s: (Label.SUCCESS if data.truth[s].state == "S" else Label.FAILURE) for s in sample_ids}


def _features(data: ExperimentData, sample_ids: Sequence[str]) -> np.ndarray:
    by_id = {kf.sample_id: kf for kfs in data.keyframes_by_episode.values() for kf in kfs}
    return np.array([by_id[s].features for s in sample_ids], dtype=float)


# ----------------------------------------------------------------------------------------------
# Audit evaluation (the ONLY place audit labels are read)
# ----------------------------------------------------------------------------------------------


def audit_ground_truth(data: ExperimentData) -> tuple[dict[str, int | None], dict[str, list[str]]]:
    targets: dict[str, int | None] = {}
    states: dict[str, list[str]] = {}
    for eid in data.ids("audit"):
        rec = data.audit_reviews.get(eid)
        if rec is None:
            raise ValueError(f"audit episode {eid!r} has no blind audit labels")
        st = []
        for kf in data.keyframes_by_episode[eid]:
            lab = rec.labels.get(kf.sample_id) if kf.auto_valid else None
            if kf.auto_valid:
                targets[kf.sample_id] = label_to_target(lab)
            st.append(label_to_state(lab) if kf.auto_valid else STATE_UNKNOWN)
        states[eid] = st
    return targets, states


def evaluate_on_audit(
    data: ExperimentData,
    scores_by_sample: Mapping[str, float],
    threshold: float,
    cfg: Mapping[str, Any],
) -> dict:
    targets, states = audit_ground_truth(data)
    per_episode: list[BinaryCounts] = []
    y_true, y_pred = [], []
    for eid in data.ids("audit"):
        et, ep = [], []
        for kf in data.keyframes_by_episode[eid]:
            if not kf.auto_valid:
                continue
            s = scores_by_sample.get(kf.sample_id)
            et.append(targets[kf.sample_id])
            ep.append(None if s is None else int(s > threshold))
        per_episode.append(count_binary(et, ep).counts)
        y_true += et
        y_pred += ep
    r = count_binary(y_true, y_pred)
    rates = binary_rates(r.counts)
    bcfg = cfg["bootstrap"]
    ci = {m: cluster_bootstrap_ci(per_episode, m, int(bcfg["n_boot"]), int(bcfg["seed"]))
          for m in ("accuracy", "fpr", "fnr")}
    score_seq = {eid: [scores_by_sample.get(kf.sample_id) if kf.auto_valid else None
                       for kf in data.keyframes_by_episode[eid]] for eid in data.ids("audit")}
    tcfg = cfg["termination"]
    replays = replay_all(score_seq, states, threshold, int(tcfg["debounce_k"]), int(tcfg["tolerance"]))
    return {
        "keyframe": {**rates, "counts": r.counts.to_dict(), "n_ref_excluded": r.n_ref_missing,
                     "n_score_missing": r.n_pred_missing, "ci95_episode_bootstrap": ci},
        "termination": summarize_replays(replays),
        # private: per-episode outcomes in sorted audit-episode order, for paired comparisons (not written out)
        "_per_episode": {"counts": per_episode, "fst": [fst_indicator(x) for x in replays]},
    }


def paired_differences(ref_eval: Mapping[str, Any], alt_eval: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict:
    """alt - ref on the same audit episodes, episode-paired bootstrap (DECISIONS.md D-11)."""
    b = cfg["bootstrap"]
    n_boot, seed = int(b["n_boot"]), int(b["seed"])
    ra, rb = ref_eval["_per_episode"], alt_eval["_per_episode"]
    out = {m: paired_cluster_bootstrap_diff(ra["counts"], rb["counts"], m, n_boot, seed)
           for m in ("accuracy", "fpr", "fnr")}
    keep = [i for i, (x, y) in enumerate(zip(ra["fst"], rb["fst"])) if x is not None and y is not None]
    as_counts = lambda xs: [BinaryCounts(fp=1) if xs[i] == 1 else BinaryCounts(tn=1) for i in keep]  # noqa: E731
    out["fst_rate"] = paired_cluster_bootstrap_diff(as_counts(ra["fst"]), as_counts(rb["fst"]), "fpr", n_boot, seed)
    out["fst_rate"]["n_episodes_paired"] = len(keep)
    return out


def global_audit_metrics(data: ExperimentData) -> dict:
    """Condition-independent audit facts: VLM zero-shot accuracy and annotator agreement."""
    targets, _ = audit_ground_truth(data)
    ids = sorted(targets)
    vlm_pred = [label_to_target(data.vlm_by_sample[s].label) if s in data.vlm_by_sample else None for s in ids]
    r = count_binary([targets[s] for s in ids], vlm_pred)
    vlm = {**binary_rates(r.counts), "counts": r.counts.to_dict(), "n_ref_excluded": r.n_ref_missing,
           "n_vlm_excluded": r.n_pred_missing}
    a, b = [], []
    for eid, rep in sorted(data.audit_repeat.items()):
        first = data.audit_reviews[eid]
        for kf in data.keyframes_by_episode[eid]:
            if not kf.auto_valid:  # auto-invalid keyframes are trivially 'agreed'; leave them out
                continue
            a.append(first.labels.get(kf.sample_id))
            b.append(rep.labels.get(kf.sample_id))
    agreement = cohen_kappa([x.value if x else None for x in a], [x.value if x else None for x in b])
    agreement["n_episodes"] = len(data.audit_repeat)
    out = {"vlm_zero_shot_vs_audit": vlm, "audit_repeat_agreement": agreement,
           "n_audit_episodes": len(data.ids("audit")),
           "audit_label_counts": _label_counts([lab for rec in data.audit_reviews.values()
                                               for lab in rec.labels.values()])}
    if data.truth is not None:  # synthetic only: how good are the simulated audit labels themselves?
        truth = _truth_labels(data, ids)
        audit_labels = {kf.sample_id: data.audit_reviews[eid].labels.get(kf.sample_id)
                        for eid in data.ids("audit") for kf in data.keyframes_by_episode[eid] if kf.auto_valid}
        out["audit_labels_vs_synthetic_truth"] = _label_quality(audit_labels, truth, ids)
    return out


def _label_counts(labels: Sequence[Label | None]) -> dict[str, int]:
    c = {"success": 0, "failure": 0, "ambiguous": 0, "invalid": 0, "none": 0}
    for lab in labels:
        c["none" if lab is None else lab.value] += 1
    return c


# ----------------------------------------------------------------------------------------------
# Conditions x replicates
# ----------------------------------------------------------------------------------------------


def run_conditions(data: ExperimentData, cfg: Mapping[str, Any],
                   scores_provider=None, label_sink=None) -> list[dict]:
    """Return one row per (replicate, condition).

    ``scores_provider(cond_labels, replicate, train_seed) -> (scores_by_sample, model_hash)`` supplies
    classifier scores for val+audit samples. For synthetic data the logistic stand-in is used; for real
    data scores must come from the LeRobot classifier on the laptop (else classifier metrics are missing).
    ``label_sink(spec, replicate, cond_labels)`` is called for every condition (e.g. to export label files).
    """
    pool = data.ids("train", "val")
    train, val = data.ids("train"), data.ids("val")
    pool_samples, train_samples, val_samples = (_valid_samples(data, x) for x in (pool, train, val))
    conditions = [ConditionSpec(c["name"], c["policy"], float(c["fraction"]), c.get("budget_unit", "episodes"))
                  for c in cfg["conditions"]]
    idle_cap = float(cfg["effort"]["idle_cap_s"])
    reference = build_condition_labels(ConditionSpec("_reference_full_review", "full", 1.0), pool,
                                       data.keyframes_by_episode, data.vlm_by_sample, data.pool_reviews,
                                       np.random.default_rng(0)).labels
    truth_pool = _truth_labels(data, pool_samples) if data.truth is not None else None
    vlm_cost = summarize_vlm_cost(pool_samples, data.vlm_by_sample)
    base = int(cfg["replicates"]["base_seed"])
    if scores_provider is None and cfg["data"]["kind"] == "synthetic":
        # the logistic stand-in needs generator features; file-based data never uses it
        scores_provider = _synthetic_scores_provider(data, cfg, train_samples)

    _check_confidence_available(data, pool_samples, conditions)
    ref_name = next((c.name for c in conditions if c.policy == "full"), None)
    rows = []
    for rep in range(int(cfg["replicates"]["n"])):
        sel_seed, train_seed = derive_seed(base, "select", rep), derive_seed(base, "train", rep)
        rep_rows = []
        for spec in conditions:
            cl = build_condition_labels(spec, pool, data.keyframes_by_episode, data.vlm_by_sample,
                                        data.pool_reviews, np.random.default_rng(sel_seed), idle_cap)
            if label_sink is not None:
                label_sink(spec, rep, cl)
            row: dict[str, Any] = {
                "condition": spec.name, "policy": spec.policy, "fraction": spec.fraction, "replicate": rep,
                "data_kind": data.data_kind.value,
                "budget": {"k": cl.k, "n_eligible_pool_episodes": cl.n_eligible_pool_episodes,
                           "realized_episode_fraction": (len(cl.selected_episodes) / cl.n_eligible_pool_episodes
                                                         if cl.n_eligible_pool_episodes else None),
                           "realized_keyframe_fraction": (sum(1 for s in pool_samples if cl.sources[s] == "human")
                                                          / len(pool_samples) if pool_samples else None),
                           **{k: v for k, v in cl.meta.items()}},
                "label_counts": cl.label_counts(),
                "label_quality_vs_full_review": _label_quality(cl.labels, reference, pool_samples),
                "effort": summarize_review_effort(cl.selected_episodes, data.pool_reviews,
                                                  data.keyframes_by_episode, data.vlm_by_sample,
                                                  float(cfg["effort"]["idle_cap_s"])),
                # every condition (incl. 100% review, where suggestions are shown) runs the VLM on the whole pool
                "vlm_cost": vlm_cost,
                "fingerprints": {"selected": _sha(cl.selected_episodes), "labels": _sha(
                    {s: (lab.value if lab else None) for s, lab in sorted(cl.labels.items())})},
                "selected_episodes": list(cl.selected_episodes),
            }
            if truth_pool is not None:
                row["label_quality_vs_synthetic_truth"] = _label_quality(cl.labels, truth_pool, pool_samples)
            if scores_provider is not None:
                scores, model_hash = scores_provider(cl, rep, train_seed)
                thr, thr_info = select_threshold(cfg["threshold"], [scores.get(s) for s in val_samples],
                                                 cl.targets(val_samples))
                row["threshold"] = {"value": thr, **thr_info}
                row["fingerprints"]["model"] = model_hash
                row["fingerprints"]["threshold"] = _sha([thr, thr_info])
                row["audit"] = evaluate_on_audit(data, scores, thr, cfg)
                sec = cfg.get("secondary_threshold")
                if sec:
                    thr2, info2 = select_threshold(sec, [scores.get(s) for s in val_samples], cl.targets(val_samples))
                    row["audit_secondary_threshold"] = {"threshold": {"value": thr2, **info2},
                                                        **evaluate_on_audit(data, scores, thr2, cfg)}
            rep_rows.append(row)
        ref = next((r for r in rep_rows if r["condition"] == ref_name), None)
        for row in rep_rows:
            if ref is not None and "audit" in row and row is not ref:
                row["paired_vs_reference"] = {"reference": ref_name, **paired_differences(ref["audit"], row["audit"], cfg)}
        for row in rep_rows:
            for key in ("audit", "audit_secondary_threshold"):
                if key in row:
                    row[key].pop("_per_episode", None)
        rows.extend(rep_rows)
    return rows


def _check_confidence_available(data: ExperimentData, pool_samples: Sequence[str],
                                conditions: Sequence[ConditionSpec], max_missing: float = 0.5) -> None:
    """Confidence-based policies silently degenerate without scores (e.g. a VLM server without logprobs):
    every keyframe would count as maximally uncertain and episodes would be ranked by LENGTH. Refuse instead."""
    if not any(c.policy in ("low_confidence", "fake_success_priority") for c in conditions):
        return
    usable = [data.vlm_by_sample[s] for s in pool_samples
              if s in data.vlm_by_sample and data.vlm_by_sample[s].label is not None]
    if not usable:
        raise ValueError("confidence-based policies need VLM suggestions, found none")
    missing = sum(1 for s in usable if s.p_success is None) / len(usable)
    if missing > max_missing:
        raise ValueError(f"{missing:.0%} of usable VLM suggestions have no p_success (no logprobs?); "
                         "low_confidence / fake_success_priority would rank episodes by length. "
                         "Enable logprobs on the VLM server or drop these conditions.")


def _synthetic_scores_provider(data: ExperimentData, cfg: Mapping[str, Any], train_samples: Sequence[str]):
    score_ids = _valid_samples(data, data.ids("val", "audit"))
    X_score = _features(data, score_ids)
    X_train_all = _features(data, train_samples)
    ccfg = cfg["classifier"]

    def provider(cl: ConditionLabels, rep: int, train_seed: int):
        t = cl.targets(train_samples)
        keep = [i for i, y in enumerate(t) if y is not None]
        model = LogisticRegressionSGD(float(ccfg["l2"]), float(ccfg["lr"]), int(ccfg["epochs"]),
                                      int(ccfg["batch_size"]), train_seed)
        model.fit(X_train_all[keep], np.array([t[i] for i in keep]))
        p = model.predict_proba(X_score)
        return {s: float(v) for s, v in zip(score_ids, p)}, model.param_hash()

    return provider


def make_scores_file_provider(path: str | Path):
    """Real data: scores computed on the laptop; rows {condition, replicate, sample_id, p_success}."""
    table: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in read_jsonl(path):
        table[(r["condition"], int(r["replicate"]))][r["sample_id"]] = float(r["p_success"])

    def provider(cl: ConditionLabels, rep: int, train_seed: int):
        key = (cl.spec.name, rep)
        if key not in table:
            raise KeyError(f"no classifier scores for condition={key[0]!r} replicate={rep}")
        return table[key], f"scores-file:{_sha(sorted(table[key].items()))}"

    return provider


# ----------------------------------------------------------------------------------------------
# Aggregation and outputs
# ----------------------------------------------------------------------------------------------


def _numeric_leaves(d: Mapping[str, Any], prefix: str = "") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_numeric_leaves(v, key + "."))
        elif isinstance(v, bool) or isinstance(v, (list, tuple, str)):
            continue
        elif v is None or isinstance(v, (int, float)):
            out[key] = None if v is None else float(v)
    return out


def aggregate(rows: Sequence[dict]) -> dict[str, dict[str, dict]]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(_numeric_leaves({k: v for k, v in r.items() if k not in ("replicate",)}))
    out: dict[str, dict[str, dict]] = {}
    for cond, leaves in by_cond.items():
        keys = sorted({k for leaf in leaves for k in leaf})
        out[cond] = {k: summarize_values([leaf.get(k) for leaf in leaves]) for k in keys}
    return out


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def run_experiment(cfg: Mapping[str, Any], out_dir: str | Path | None = None, data: ExperimentData | None = None,
                   label_sink=None) -> dict:
    from hilreview.experiments.report import render_report  # local import to avoid a cycle

    data = data if data is not None else prepare_data(cfg)
    scores_provider = None
    if cfg["data"]["kind"] == "files" and cfg["data"]["files"].get("classifier_scores"):
        scores_provider = make_scores_file_provider(cfg["data"]["files"]["classifier_scores"])
    rows = run_conditions(data, cfg, scores_provider, label_sink)
    if data.data_kind is DataKind.SYNTHETIC and cfg["data"]["kind"] == "synthetic":
        clf_source = "logistic stand-in on synthetic features (SIMULATED)"
    elif scores_provider is not None:
        clf_source = f"scores file: {cfg['data']['files']['classifier_scores']}"
    else:
        clf_source = "MISSING (no classifier scores provided)"
    sources = {
        "data_kind": data.data_kind.value,
        "vlm_models": sorted({s.model_id for s in data.vlm_by_sample.values()}),
        "vlm_latency_sources": sorted({s.latency_source for s in data.vlm_by_sample.values()}),
        "vlm_confidence_sources": sorted({s.confidence_source for s in data.vlm_by_sample.values()}),
        "pool_review_timing_sources": sorted({r.timing_source for r in data.pool_reviews.values()}),
        "audit_timing_sources": sorted({r.timing_source for r in data.audit_reviews.values()}),
        "reviewer_ids": sorted({r.reviewer_id for r in (*data.pool_reviews.values(), *data.audit_reviews.values())}),
        "classifier": clf_source,
    }
    summary = {"conditions": aggregate(rows), "global": global_audit_metrics(data), "sources": sources,
               "split_counts": {s: sum(1 for v in data.split.values() if v == s)
                                for s in ("train", "val", "audit", "excluded")}}
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    provenance = {
        "experiment_name": cfg["experiment_name"],
        "data_kind": data.data_kind.value,
        "evidence_status": ("SYNTHETIC: pipeline validation only, not evidence about real systems"
                            if data.data_kind is DataKind.SYNTHETIC else "REAL DATA"),
        "config_hash": config_hash(cfg_clean),
        "config_path": cfg.get("_config_path"),
        "created_at_utc": utc_now_iso(),
        "git": git_info(),
        "environment": environment_info(),
        "replicates": int(cfg["replicates"]["n"]),
        "base_seed": int(cfg["replicates"]["base_seed"]),
    }
    result = {"provenance": provenance, "rows": rows, "summary": summary}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
        (out / "config.resolved.yaml").write_text(yaml.safe_dump(cfg_clean, sort_keys=True))
        (out / "splits.json").write_text(json.dumps({"split_spec": cfg["splits"], "counts": summary["split_counts"],
                                                     "episode_split": dict(sorted(data.split.items()))},
                                                    indent=2, sort_keys=True))
        (out / "per_replicate.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        flat = [_flatten({k: v for k, v in r.items() if k != "selected_episodes"}) for r in rows]
        cols = sorted({c for f in flat for c in f})
        with open(out / "per_replicate.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(flat)
        (out / "report.md").write_text(render_report(provenance, summary, cfg_clean))
    return result
