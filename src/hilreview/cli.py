"""Command-line interface: `hilreview <subcommand>` (or `python -m hilreview.cli <subcommand>`).

Subcommands that only touch files run anywhere. Subcommands marked [laptop] need LeRobot/torch/a local VLM
server and are UNVERIFIED in the sandbox. No subcommand moves a robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from hilreview.data.io import append_jsonl, read_jsonl, write_jsonl
from hilreview.data.schema import DataKind, Episode, Keyframe, ReviewRecord, VLMSuggestion


def _load_eps_kfs(episodes: str, keyframes: str):
    eps = read_jsonl(episodes, Episode.from_dict)
    kfs: dict[str, list[Keyframe]] = {}
    for kf in read_jsonl(keyframes, Keyframe.from_dict):
        kfs.setdefault(kf.episode_id, []).append(kf)
    for v in kfs.values():
        v.sort(key=lambda k: k.frame_index)
    return eps, kfs


def _split_map(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["episode_split"]


def cmd_run(a) -> int:
    from hilreview.experiments.config import load_config
    from hilreview.experiments.runner import prepare_data, run_experiment
    from hilreview.robot.lerobot_io import write_condition_label_file

    cfg = load_config(a.config)
    out = Path(a.out or Path(cfg["output_root"]) / cfg["data"]["kind"] / cfg["experiment_name"])
    data = prepare_data(cfg)
    sink = None
    if a.export_labels:
        pool = data.ids("train", "val")
        kfs = [kf for eid in pool for kf in data.keyframes_by_episode[eid]]

        def sink(spec, rep, cl):
            write_condition_label_file(out / "labels" / f"{spec.name}_rep{rep}.jsonl", kfs, cl.labels, cl.sources,
                                       spec.name, rep)

    res = run_experiment(cfg, out, data=data, label_sink=sink)
    kind_ko = {"synthetic": "합성(가짜) 데이터", "real": "실제 데이터"}.get(res['provenance']['data_kind'], res['provenance']['data_kind'])
    print(f"[{kind_ko}] 결과 저장 위치: {out}")
    print(f"리포트: {out / 'report.md'}")
    return 0


def cmd_split(a) -> int:
    from hilreview.data.splits import SplitSpec, assign_splits, check_split_integrity
    from hilreview.utils.provenance import sha256_file, utc_now_iso

    eps = read_jsonl(a.episodes, Episode.from_dict)
    spec = SplitSpec(a.audit_fraction, a.val_fraction, a.group_by, tuple(a.stratify_by.split(",")) if a.stratify_by
                     else (), a.seed)
    split = assign_splits(eps, spec)
    counts = check_split_integrity(eps, split, group_by=spec.group_by)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f"이미 고정된 데이터 나누기 파일이 있어 덮어쓰지 않습니다: {out} (--overwrite는 라벨링을 시작하기 전에만 쓰세요)")
        return 2
    out.write_text(json.dumps({"split_spec": spec.to_dict(), "counts": counts, "created_at_utc": utc_now_iso(),
                               "episodes_sha256": sha256_file(a.episodes), "episode_split": split},
                              indent=2, sort_keys=True))
    print(f"데이터 나누기 고정 완료: {counts} -> {out}")
    return 0


def cmd_vlm_label(a) -> int:
    from hilreview.labeling.vlm import OpenAICompatConfig, OpenAICompatVLMLabeler

    eps, kfs = _load_eps_kfs(a.episodes, a.keyframes)
    wanted = None
    if a.split_file:
        split = _split_map(a.split_file)
        wanted = set(a.splits.split(","))
        eps = [e for e in eps if split.get(e.episode_id) in wanted]
    out = Path(a.out)
    done: set[str] = set()
    if out.exists():
        latest = {r["sample_id"]: r for r in read_jsonl(out)}  # last record per sample wins (also in the loader)
        done = {sid for sid, r in latest.items() if not (a.retry_errors and r["status"] != "ok")}
    kinds = {e.data_kind for e in eps}
    if len(kinds) != 1:
        print(f"에피소드들의 데이터 종류가 섞여 있습니다: {sorted(k.value for k in kinds)}")
        return 2
    cfg = OpenAICompatConfig(base_url=a.base_url, model=a.model, api_key_env=a.api_key_env, timeout_s=a.timeout,
                             camera_order=tuple(a.cameras.split(",")) if a.cameras else ())
    lab = OpenAICompatVLMLabeler(cfg, task=a.task, data_kind=kinds.pop(), allow_remote_host=a.allow_remote_host)
    todo = [kf for e in eps if e.eligible for kf in kfs.get(e.episode_id, []) if kf.auto_valid
            and kf.sample_id not in done]
    if a.limit is not None:
        todo = todo[: a.limit]
    n_err = 0
    for i, kf in enumerate(todo, 1):
        s = lab.suggest_one(kf)
        n_err += s.status != "ok"
        append_jsonl(out, s)
        if i % 50 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}개 라벨링 (오류 {n_err}개)")
    print(f"완료: 새 VLM 제안 {len(todo)}개 -> {out}")
    return 0


def cmd_review(a) -> int:
    from hilreview.review.session import ReviewSession, make_display

    eps, kfs = _load_eps_kfs(a.episodes, a.keyframes)
    split = _split_map(a.split_file)
    target = {"pool_review": ("train", "val"), "audit": ("audit",), "audit_repeat": ("audit",)}[a.purpose]
    eps = [e for e in eps if e.eligible and split.get(e.episode_id) in target]
    if a.purpose == "audit_repeat" and a.repeat_fraction < 1.0:
        from hilreview.utils.numeric import round_half_up
        from hilreview.utils.rng import make_rng

        ids = sorted(e.episode_id for e in eps)
        n = round_half_up(a.repeat_fraction * len(ids))
        keep = {ids[i] for i in make_rng(a.order_seed, "audit_repeat_subset").permutation(len(ids))[:n]}
        eps = [e for e in eps if e.episode_id in keep]
    vlm = None
    if a.purpose == "pool_review":
        if not a.vlm:
            print("pool_review(보조 검토)에는 --vlm 파일이 필요합니다")
            return 2
        vlm = {s.sample_id: s for s in read_jsonl(a.vlm, VLMSuggestion.from_dict)}
    kinds = {e.data_kind for e in eps}
    if len(kinds) != 1:
        print(f"에피소드들의 데이터 종류가 섞여 있습니다: {sorted(k.value for k in kinds)}")
        return 2
    kind = kinds.pop()  # review records inherit the episodes' data kind (demo data stays 'synthetic')
    if kind is DataKind.SYNTHETIC:
        print("참고: 합성(가짜) 에피소드입니다. 여기서 단 라벨과 시간은 연습용이며 연구 데이터가 아닙니다.")
    sess = ReviewSession(eps, kfs, a.out, a.reviewer_id, a.purpose, vlm, order_seed=a.order_seed, data_kind=kind,
                         session_id=a.session_id, display_fn=make_display(Path(a.filmstrip_dir), a.viewer))
    recs = sess.run(max_episodes=a.max_episodes)
    print(f"검토 {len(recs)}개 저장 완료: {a.out}")
    return 0


def cmd_schedule(a) -> int:
    from hilreview.experiments.schedule import make_blinded_schedule

    sched, key = make_blinded_schedule(a.conditions.split(","), a.replicates, a.seed)
    for path, obj in ((a.out_schedule, sched), (a.out_key, key)):
        if Path(path).exists():
            print(f"이미 있는 파일이라 덮어쓰지 않습니다: {path}")
            return 2
        Path(path).write_text(json.dumps(obj, indent=2))
    print(f"실행 순서표 ({len(sched)}회) -> {a.out_schedule}\n조건 key(봉인) -> {a.out_key} "
          "(모든 실행과 성공 판단이 끝날 때까지 열지 마세요)")
    return 0


def cmd_online_summary(a) -> int:
    from hilreview.evaluation.online import (OnlineEpisodeRecord, RunRecord, compare_to_reference,
                                             summarize_conditions)

    eps = read_jsonl(a.episodes, OnlineEpisodeRecord.from_dict)
    runs = read_jsonl(a.runs, RunRecord.from_dict)
    key = {k["run_id"]: k for k in json.loads(Path(a.key).read_text())}
    out = summarize_conditions(eps, runs, key)
    if a.reference:
        comp = compare_to_reference(out, a.reference, a.delta, a.conf)
        out = {"conditions": out, "comparisons_D30": comp}
        ko = {"noninferior": "비열등", "inferior": "열등", "inconclusive": "판단 불가", "insufficient_runs": "실행 부족"}
        for cond, c in comp.items():
            p, s = c["primary"], c["sensitivity_failed_runs_as_zero"]
            bound = "" if p["lower"] is None else f" (차이 {p['diff']:+.3f}, 단측 {p['conf']:.0%} 구간 [{p['lower']:+.3f}, {p['upper']:+.3f}], -δ={-p['delta']:+.2f})"
            print(f"{cond} − {a.reference}: {ko[p['decision']]}{bound} | 실패 실행 0 처리 시: {ko[s['decision']]}")
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(f"실제 로봇 결과 요약 -> {a.out}")
    return 0


def cmd_vlm_check(a) -> int:
    """One request to a LOCAL VLM server: reachable? answer parsed? logprobs (confidence) returned? latency?"""
    import json as _json
    import tempfile
    import urllib.request

    from hilreview.data.schema import Keyframe, make_sample_id
    from hilreview.labeling.vlm import OpenAICompatConfig, OpenAICompatVLMLabeler
    from hilreview.review.demo import _render

    cfg = OpenAICompatConfig(base_url=a.base_url, model=a.model, timeout_s=a.timeout)
    try:
        lab = OpenAICompatVLMLabeler(cfg, task="put the red cube into the blue container",
                                     allow_remote_host=a.allow_remote_host)
    except ValueError as exc:
        print(f"중단: {exc}")
        return 2
    try:
        with urllib.request.urlopen(a.base_url.rstrip("/") + "/models", timeout=a.timeout) as r:
            models = [m.get("id") for m in _json.loads(r.read().decode()).get("data", [])]
        print(f"[1/3] 서버 응답 OK. 제공 모델: {models}")
        if a.model not in models:
            print(f"      주의: --model {a.model!r}이 목록에 없습니다. 목록의 이름을 --model로 주세요.")
    except Exception as exc:  # noqa: BLE001
        print(f"[1/3] 서버에 연결하지 못했습니다: {type(exc).__name__}: {exc}")
        return 1
    results = []
    with tempfile.TemporaryDirectory() as td:
        for i, (state, hover) in enumerate([("S", False), ("F", True), ("F", False)]):
            img = f"{td}/front_{i}.png"
            _render(state, hover, False, 0.5).save(img)
            kf = Keyframe(make_sample_id("vlmcheck", i), "vlmcheck", i, float(i), {"front": img})
            results.append((state, hover, lab.suggest_one(kf)))
    ok = [s for _, _, s in results if s.status == "ok"]
    print(f"[2/3] 응답 해석: {len(ok)}/3 성공")
    for state, hover, s in results:
        truth = "성공 장면" if state == "S" else ("용기 위에 든 장면(가짜 성공 유도)" if hover else "실패 장면")
        print(f"      {truth:22s} -> label={s.label.value if s.label else None}, p_success={s.p_success}, "
              f"status={s.status}, {s.latency_s:.2f}s, raw={str(s.raw_response)[:40]!r}")
    has_lp = any(s.p_success is not None for s in ok)
    print(f"[3/3] 확신 점수(logprobs): {'있음' if has_lp else '없음'}"
          + ("" if has_lp else " -> '확신 낮은 것 우선'·'가짜 성공 우선' 정책을 쓸 수 없습니다 (무작위/A/B/C는 가능)"))
    print("참고: 그림은 만화 같은 연습용입니다. 여기서의 정답률은 실제 성능과 무관합니다.")
    return 0 if ok else 1


def cmd_log_episodes(a) -> int:
    from hilreview.evaluation.episode_logger import EpisodeOutcomeLogger

    if a.data_kind == "synthetic":
        print("참고: 연습 모드(synthetic)입니다. 이 기록은 연구 데이터가 아닙니다.")
    logger = EpisodeOutcomeLogger(a.run_id, a.phase, a.out, a.runs_out, a.fps, DataKind(a.data_kind))
    rr = logger.run()
    print(f"기록 파일: {a.out}" + (f" | 실행 상태: {rr.status} -> {a.runs_out}" if rr else ""))
    return 0


def cmd_make_demo_data(a) -> int:
    from hilreview.review.demo import make_demo_data

    paths = make_demo_data(Path(a.out), n_episodes=a.n_episodes, seed=a.seed)
    print("합성(가짜) 연습 데이터 (검토 도구 연습용):")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


def cmd_export_keyframes(a) -> int:  # pragma: no cover - laptop only
    from hilreview.robot.lerobot_io import export_keyframes

    meta = json.loads(Path(a.episode_meta).read_text())  # {"<episode_index>": {"session_id":..,"collection_mode":..}}
    session_of = {int(k): v["session_id"] for k, v in meta.items()}
    mode_of = {int(k): v["collection_mode"] for k, v in meta.items()}
    inel = {int(k): v["ineligible_reason"] for k, v in meta.items() if v.get("ineligible_reason")}
    paths = export_keyframes(a.repo_id, a.root, a.out, a.stride, session_of, mode_of,
                             a.cameras.split(",") if a.cameras else None, inel)
    print(f"저장 완료: {paths}")
    return 0


def cmd_build_reward_dataset(a) -> int:  # pragma: no cover - laptop only
    from hilreview.robot.lerobot_io import build_reward_dataset

    n = build_reward_dataset(a.src_repo_id, a.src_root, a.episodes, a.labels, a.out_repo_id, a.out_root, a.mode)
    print(f"라벨 붙은 프레임 {n}개 저장: {a.out_root}")
    return 0


def cmd_score(a) -> int:  # pragma: no cover - laptop only
    from hilreview.robot.lerobot_io import score_keyframes

    split = _split_map(a.split_file)
    ids = [e for e, s in split.items() if s in a.splits.split(",")]
    n = score_keyframes(a.classifier, a.repo_id, a.root, a.episodes, a.keyframes, ids, a.out, a.condition,
                        a.replicate, a.input_mode, a.device)
    print(f"키프레임 {n}개 점수 계산 -> {a.out}")
    return 0


def _demo_ids(a) -> list[str]:
    from hilreview.robot.demo_relabel import select_demo_episodes

    ids = select_demo_episodes(read_jsonl(a.episodes, Episode.from_dict), _split_map(a.split_file))
    if not ids:
        raise SystemExit("시연 에피소드가 없습니다 (train 분할의 success_attempt 에피소드가 0개)")
    return ids


def cmd_score_frames(a) -> int:  # pragma: no cover - laptop only
    from hilreview.robot.demo_relabel import score_demo_frames

    ids = _demo_ids(a)
    n = score_demo_frames(a.classifier, a.repo_id, a.root, a.episodes, ids, a.out, a.condition, a.replicate,
                          a.input_mode, a.device, a.batch_size)
    print(f"[{a.condition} rep{a.replicate}] 시연 에피소드 {len(ids)}개, 프레임 {n}개 점수 계산 -> {a.out}")
    return 0


def cmd_relabel_demos(a) -> int:  # pragma: no cover - laptop only (tested via tests/test_demo_relabel_lerobot.py)
    from hilreview.robot.demo_relabel import write_relabeled_demo_dataset

    ids = _demo_ids(a)
    rows = write_relabeled_demo_dataset(a.src_repo_id, a.src_root, a.episodes, ids, a.scores, a.out_repo_id,
                                        a.out_root, a.manifest, a.threshold, a.success_reward, a.image_storage)
    trig = sum(r["trigger_frame"] is not None for r in rows)
    kept = sum(r["n_keep"] for r in rows)
    print(f"시연 에피소드 {len(rows)}개 재라벨: 분류기 성공 판정 {trig}개, 판정 없음 {len(rows) - trig}개, "
          f"남긴 프레임 {kept}개 -> {a.out_root}")
    print(f"기록 파일: {a.manifest}")
    return 0


def cmd_extract_interventions(a) -> int:  # pragma: no cover - needs wandb (tested in tests/test_interventions_lerobot.py)
    from hilreview.evaluation.interventions import attach_to_records, episodes_from_history, read_wandb_history
    from hilreview.evaluation.online import OnlineEpisodeRecord

    eps = episodes_from_history(read_wandb_history(a.wandb_run), a.first_step_offset)
    write_jsonl(a.out, [e.to_dict() for e in eps])
    steps = sum(e.intervention_steps for e in eps)
    print(f"actor 에피소드 {len(eps)}개, 개입 에피소드 {sum(e.intervention_flag for e in eps)}개, "
          f"개입 스텝 {steps}개 = {steps / a.fps:.1f}초 (fps {a.fps}) -> {a.out}")
    if a.episodes:
        if not (a.run_id and a.episodes_out):
            raise SystemExit("--episodes를 주면 --run-id와 --episodes-out도 필요합니다")
        recs = read_jsonl(a.episodes, OnlineEpisodeRecord.from_dict)
        out, rep = attach_to_records(recs, eps, a.run_id, a.fps)
        write_jsonl(a.episodes_out, out)
        print(f"사람 기록에 개입 스텝 채움 -> {a.episodes_out}")
        if rep["human_vs_actor_flag_disagreements"]:
            print(f"주의: 사람이 표시한 개입 여부와 actor 기록이 다른 에피소드 {rep['human_vs_actor_flag_disagreements']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hilreview", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run conditions x replicates from a YAML config (synthetic or real labels)")
    r.add_argument("--config", required=True)
    r.add_argument("--out")
    r.add_argument("--export-labels", action="store_true", help="write per-condition label JSONL for LeRobot")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("split", help="assign and FREEZE the episode-level train/val/audit split (real data)")
    s.add_argument("--episodes", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--audit-fraction", type=float, default=0.25)
    s.add_argument("--val-fraction", type=float, default=0.2)
    s.add_argument("--group-by", default="episode_id", choices=["episode_id", "session_id"])
    s.add_argument("--stratify-by", default="session_id,collection_mode")
    s.add_argument("--seed", type=int, default=20261017)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_split)

    v = sub.add_parser("vlm-label", help="[laptop] label keyframes with a LOCAL OpenAI-compatible VLM server")
    v.add_argument("--episodes", required=True)
    v.add_argument("--keyframes", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--task", required=True, help='e.g. "put the red cube into the bowl"')
    v.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    v.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    v.add_argument("--api-key-env", default=None, help="NAME of an env var with a key (local servers need none)")
    v.add_argument("--timeout", type=float, default=60.0)
    v.add_argument("--cameras", default=None, help="comma-separated camera keys in prompt order")
    v.add_argument("--split-file")
    v.add_argument("--splits", default="train,val,audit")
    v.add_argument("--limit", type=int, default=None, help="label at most N new keyframes (pilot runs)")
    v.add_argument("--retry-errors", action="store_true", help="re-label samples whose latest record failed")
    v.add_argument("--allow-remote-host", action="store_true",
                   help="allow a non-local/non-LAN server (sends private robot images off your machines)")
    v.set_defaults(fn=cmd_vlm_label)

    rv = sub.add_parser("review", help="interactive human review / blind audit labeling (measures time)")
    rv.add_argument("--episodes", required=True)
    rv.add_argument("--keyframes", required=True)
    rv.add_argument("--split-file", required=True)
    rv.add_argument("--purpose", required=True, choices=["pool_review", "audit", "audit_repeat"])
    rv.add_argument("--out", required=True)
    rv.add_argument("--reviewer-id", required=True)
    rv.add_argument("--vlm", help="vlm_suggestions.jsonl (pool_review only)")
    rv.add_argument("--viewer", default="print", choices=["print", "xdg-open", "open"])
    rv.add_argument("--filmstrip-dir", default="review_filmstrips")
    rv.add_argument("--order-seed", type=int, default=77)
    rv.add_argument("--repeat-fraction", type=float, default=0.3)
    rv.add_argument("--max-episodes", type=int, default=None)
    rv.add_argument("--session-id", default="")
    rv.set_defaults(fn=cmd_review)

    sc = sub.add_parser("schedule", help="blinded, block-randomized real-robot run schedule")
    sc.add_argument("--conditions", required=True)
    sc.add_argument("--replicates", type=int, required=True)
    sc.add_argument("--seed", type=int, required=True)
    sc.add_argument("--out-schedule", required=True)
    sc.add_argument("--out-key", required=True)
    sc.set_defaults(fn=cmd_schedule)

    o = sub.add_parser("online-summary", help="unblind and summarize real-robot episode logs")
    o.add_argument("--episodes", required=True)
    o.add_argument("--runs", required=True)
    o.add_argument("--key", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--reference", default=None,
                   help="D-30 기준 조건 이름 (예: A_full_review). 주면 X − A Welch 비열등 판단을 함께 계산")
    o.add_argument("--delta", type=float, default=0.20, help="비열등 한계 δ (D-30 사전 등록값 0.20)")
    o.add_argument("--conf", type=float, default=0.95, help="단측 신뢰수준 (D-30 사전 등록값 0.95)")
    o.set_defaults(fn=cmd_online_summary)

    vc = sub.add_parser("vlm-check", help="[노트북] 로컬 VLM 서버 점검: 연결, 응답 해석, 확신 점수(logprobs), 속도")
    vc.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    vc.add_argument("--model", required=True)
    vc.add_argument("--timeout", type=float, default=120.0)
    vc.add_argument("--allow-remote-host", action="store_true")
    vc.set_defaults(fn=cmd_vlm_check)

    lg = sub.add_parser("log-episodes", help="실제 로봇 실행 중 에피소드 결과 기록 (글자 2개 입력)")
    lg.add_argument("--run-id", required=True, help="블라인드 순서표의 실행 ID (예: run-1a2b3c4d)")
    lg.add_argument("--phase", required=True, choices=["train", "eval"])
    lg.add_argument("--out", required=True, help="online_episodes.jsonl")
    lg.add_argument("--runs-out", default=None, help="runs.jsonl (train 단계에서 실행 상태 기록)")
    lg.add_argument("--fps", type=float, default=None)
    lg.add_argument("--data-kind", default="real", choices=["real", "synthetic"], help="synthetic = 연습용")
    lg.set_defaults(fn=cmd_log_episodes)

    d = sub.add_parser("make-demo-data", help="SYNTHETIC images+files to practise the review workflow")
    d.add_argument("--out", required=True)
    d.add_argument("--n-episodes", type=int, default=12)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_make_demo_data)

    ek = sub.add_parser("export-keyframes", help="[laptop, UNVERIFIED] LeRobot dataset -> keyframe PNGs + JSONL")
    ek.add_argument("--repo-id", required=True)
    ek.add_argument("--root", default=None)
    ek.add_argument("--out", required=True)
    ek.add_argument("--stride", type=int, default=5)
    ek.add_argument("--episode-meta", required=True)
    ek.add_argument("--cameras", default=None)
    ek.set_defaults(fn=cmd_export_keyframes)

    b = sub.add_parser("build-reward-dataset", help="[laptop, UNVERIFIED] condition labels -> new LeRobot dataset")
    b.add_argument("--src-repo-id", required=True)
    b.add_argument("--src-root", default=None)
    b.add_argument("--episodes", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--out-repo-id", required=True)
    b.add_argument("--out-root", required=True)
    b.add_argument("--mode", default="keyframes_only", choices=["keyframes_only", "hold_previous", "nearest"])
    b.set_defaults(fn=cmd_build_reward_dataset)

    sk = sub.add_parser("score", help="[laptop, UNVERIFIED] score val/audit keyframes with a trained classifier")
    sk.add_argument("--classifier", required=True)
    sk.add_argument("--repo-id", required=True)
    sk.add_argument("--root", default=None)
    sk.add_argument("--episodes", required=True)
    sk.add_argument("--keyframes", required=True)
    sk.add_argument("--split-file", required=True)
    sk.add_argument("--splits", default="val,audit")
    sk.add_argument("--out", required=True)
    sk.add_argument("--condition", required=True)
    sk.add_argument("--replicate", type=int, required=True)
    sk.add_argument("--input-mode", default="runtime", choices=["runtime", "train"])
    sk.add_argument("--device", default="cuda")
    sk.set_defaults(fn=cmd_score)

    sf = sub.add_parser("score-frames", help="[laptop] D-29: 시연 에피소드의 모든 프레임을 조건의 분류기로 채점")
    sf.add_argument("--classifier", required=True)
    sf.add_argument("--repo-id", required=True)
    sf.add_argument("--root", default=None)
    sf.add_argument("--episodes", required=True)
    sf.add_argument("--split-file", required=True)
    sf.add_argument("--out", required=True)
    sf.add_argument("--condition", required=True)
    sf.add_argument("--replicate", type=int, required=True)
    sf.add_argument("--input-mode", default="runtime", choices=["runtime", "train"],
                    help="runtime = 온라인 실행과 같은 입력 (D-29 기본값)")
    sf.add_argument("--device", default="cuda")
    sf.add_argument("--batch-size", type=int, default=64)
    sf.set_defaults(fn=cmd_score_frames)

    rd = sub.add_parser("relabel-demos", help="[laptop] D-29/D-35: 시연 데이터 보상을 조건의 분류기 점수로 다시 매긴 새 데이터셋")
    rd.add_argument("--src-repo-id", required=True)
    rd.add_argument("--src-root", default=None)
    rd.add_argument("--episodes", required=True)
    rd.add_argument("--split-file", required=True)
    rd.add_argument("--scores", required=True, help="score-frames 출력 (한 조건·한 반복)")
    rd.add_argument("--out-repo-id", required=True)
    rd.add_argument("--out-root", required=True)
    rd.add_argument("--manifest", required=True, help="에피소드별 판정 위치·사람 버튼 위치 기록 (JSON)")
    rd.add_argument("--threshold", type=float, default=0.5, help="온라인 success_threshold와 같아야 함")
    rd.add_argument("--success-reward", type=float, default=1.0, help="온라인 success_reward와 같아야 함")
    rd.add_argument("--image-storage", default="image", choices=["image", "video"],
                    help="image = 무손실 저장 (영상 재압축 없음, 기본값)")
    rd.set_defaults(fn=cmd_relabel_demos)

    xi = sub.add_parser("extract-interventions",
                        help="[laptop] D-26/D-36: learner의 오프라인 wandb 기록 -> 에피소드별 개입 스텝·전체 스텝")
    xi.add_argument("--wandb-run", required=True, help="<output_dir>/wandb/offline-run-* 폴더 또는 run-*.wandb 파일")
    xi.add_argument("--fps", type=float, required=True, help="HIL-SERL env.fps (개입 시간 = 스텝 / fps)")
    xi.add_argument("--out", required=True, help="actor 에피소드 JSONL")
    xi.add_argument("--first-step-offset", type=int, default=0, help="이 실행의 첫 에피소드 시작 스텝 (새 실행이면 0)")
    xi.add_argument("--episodes", default=None, help="(선택) log-episodes 기록: 이 실행의 train 에피소드에 채워 넣음")
    xi.add_argument("--run-id", default=None)
    xi.add_argument("--episodes-out", default=None, help="채운 기록을 쓸 새 파일 (원본은 그대로)")
    xi.set_defaults(fn=cmd_extract_interventions)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
