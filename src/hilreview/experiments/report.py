"""Markdown report that separates measured, simulated and missing results (Korean or English).

Rules: a SYNTHETIC banner whenever data_kind == synthetic; every metric family states its source;
metrics that require real-robot runs are listed as MISSING (never filled with placeholders);
replicate counts are shown; no significance claims are generated.
Language: cfg["report_lang"] = "ko" (default) or "en". Only the wording changes, never the numbers.
"""

from __future__ import annotations

from typing import Any, Mapping

TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "title": "# Experiment report: {name}\n",
        "banner": (
            "> **WARNING: SYNTHETIC DATA.** The episodes below are synthetic (generated, not recorded on a robot).\n"
            "> Simulator parameters (mock VLM error rates, simulated reviewer accuracy/speed) are **invented**.\n"
            "> These results only check that the pipeline is wired correctly. They are **not evidence** about real\n"
            "> robots, real VLMs (e.g. Qwen3-VL), real human review time, or HIL-SERL success rates.\n"
        ),
        "prov_h": "## Provenance\n", "field_value": "| field | value |",
        "p_kind": "data kind", "p_hash": "config hash", "p_git": "git commit", "p_created": "created (UTC)",
        "p_reps": "replicates", "p_reps_val": "{n} (base seed {seed})", "p_py": "python / numpy",
        "split_h": "## Data and splits\n",
        "split_txt": ("Episode-level split (group_by = `{group_by}`, stratified by `{strat}`, seed {seed}): "
                      "train {train}, val {val}, audit {audit}, excluded (ineligible) {excluded}. "
                      "The same audit episodes are used for every condition and replicate.\n"),
        "glob_h": "## Condition-independent audit facts\n", "q_v_s": "| quantity | value | source |",
        "vlm_acc": "VLM zero-shot accuracy on audit keyframes",
        "vlm_acc_val": "{acc} (FPR {fpr}, FNR {fnr}, n={n}, {vx} VLM-excluded, {ax} audit-excluded)",
        "mock_vlm": "mock VLM (SIMULATED)", "kappa": "Blind audit repeat agreement (Cohen's kappa, 4 labels)",
        "kappa_val": "{k} (observed {o}, {n} keyframes, {e} episodes)", "audit_timing": "audit timing: {t}",
        "audit_counts": "Audit label counts", "reviewers": "reviewers {r}",
        "truth_row": "Audit labels vs synthetic ground truth (accuracy)", "truth_src": "synthetic only: real data has no oracle",
        "cond_h": "## Per-condition results (mean ± SD over replicates; see the note on uncertainty)\n",
        "thr_txt": ("Primary threshold: `{thr}` (pre-registered; never tuned on audit). Termination replay: "
                    "debounce_k={dk}, tolerance={tol} keyframes.\n"),
        "cost_h": "### Labeling cost (human review time source: **{src}**)\n",
        "cost_txt": ("Time capped = each episode's time capped at {cap:.0f} s (idle guard); both are shown. "
                     "Equal episode budgets are NOT equal time budgets: compare the time columns.\n"),
        "cost_cols": ("| condition | budget unit | episodes reviewed (k / N) | keyframe fraction reviewed | "
                      "time fraction used (capped) | human time [min] | human time capped [min] | corrections | "
                      "VLM calls |"),
        "unit_episodes": "episodes", "unit_seconds": "time",
        "lq_h": "### Training-label quality on the labeling pool\n",
        "lq_txt": ("Reference = labels from the 100% human review (itself imperfect). Ambiguous/invalid/no-label "
                   "keyframes are excluded from training and counted.\n"),
        "lq_cols": "| condition | accuracy vs full review | FPR (fake success) | FNR | excluded keyframes |",
        "lq_truth_col": " accuracy vs synthetic truth |", "reference": "(reference)",
        "clf_h": "### Reward classifier on the blind audit set\n",
        "clf_src": "Classifier source: **{src}**. Audit set: {n} episodes.",
        "unc_txt": ("**How to read the uncertainty.** `±` is the SD across replicates. The audit set is fixed, so it "
                    "only reflects training/selection randomness. `[lo, hi]` is the 95% episode-cluster bootstrap "
                    "interval (Wilson for the termination rate), averaged over replicates. It reflects how few audit "
                    "episodes there are and is usually much wider. Compare conditions with the paired table below, "
                    "not with the `±` values."),
        "clf_cols": ("| condition | keyframe accuracy | FPR | FNR | precision | false-success termination rate | "
                     "missed-success rate | missing scores |"),
        "paired_h": "#### Paired differences vs `{ref}` (same audit episodes; episode-paired bootstrap)\n",
        "paired_txt": ("Entries: difference (condition minus reference), mean over replicates, with the "
                       "replicate-averaged 95% interval. An interval that contains 0 means no difference was resolved "
                       "with this audit set.\n"),
        "paired_cols": "| condition | Δ keyframe accuracy | Δ FPR | Δ FNR | Δ false-success termination rate |",
        "proxy_txt": ("The false-success termination rate is an offline **proxy**: classifier scores are replayed over "
                      "recorded audit episodes at keyframe resolution, which is coarser than the online control rate. "
                      "It is not an online measurement."),
        "no_scores": "**MISSING**: no classifier scores were provided for this run.",
        "mp_h": "## Metric provenance\n", "mp_cols": "| metric family | source |",
        "mp_data": "Data", "mp_vlm": "VLM suggestions",
        "mp_vlm_val": "models {m}, latency {l}, confidence {c}",
        "mp_pool": "Pool review timing", "mp_pool_val": "{t} (reviewers {r})", "mp_audit": "Audit labeling timing",
        "mp_clf": "Reward classifier", "mp_robot": "| Real-robot metrics | **MISSING** (see below) |",
        "miss_h": "## Missing metrics (not estimated, not imputed)\n", "miss_item": "- **{name}**: MISSING, {why}.",
        "missing": [
            ("Real-robot task success rate (human-judged)", "requires blinded real-robot evaluation episodes"),
            ("Online false-success termination rate", "requires human-judged outcomes of classifier-terminated episodes"),
            ("Human intervention count / time during online RL", "requires LeRobot actor logs from real runs"),
            ("Training time, wall-clock time, run failure rate", "requires real HIL-SERL runs"),
            ("Labels reviewed per training hour", "requires real training wall-clock time"),
            ("Between-seed variance of robot success", "requires >= 2 real seeds per condition"),
        ],
        "guard_h": "## Interpretation guardrails\n",
        "guard_reps": "- {n} replicates per condition. Differences are descriptive; no significance test was run.",
        "guard_robot": ("- Better label or classifier accuracy does not imply better robot success; only blinded "
                        "real-robot evaluation can show that."),
        "guard_syn": ("- Differences between conditions here follow from the simulator's invented error and timing "
                      "parameters. Changing those parameters changes the conclusions."),
        "undefined": "undefined", "none": "none",
        "values": {},
    },
    "ko": {
        "title": "# 실험 리포트: {name}\n",
        "banner": (
            "> **경고: 합성(가짜) 데이터.** 아래 에피소드는 로봇에서 녹화한 것이 아니라 컴퓨터가 만든 것입니다.\n"
            "> 시뮬레이터 수치(가짜 VLM 오류율, 가짜 검토자의 정확도·속도)는 **지어낸 값**입니다.\n"
            "> 이 결과는 프로그램이 제대로 연결됐는지만 확인합니다. 실제 로봇, 실제 VLM(예: Qwen3-VL),\n"
            "> 실제 사람 검토 시간, HIL-SERL 성공률에 대한 **증거가 아닙니다**.\n"
        ),
        "prov_h": "## 출처 기록\n", "field_value": "| 항목 | 값 |",
        "p_kind": "데이터 종류", "p_hash": "설정 해시", "p_git": "git 커밋", "p_created": "생성 시각 (UTC)",
        "p_reps": "반복 실행", "p_reps_val": "{n}회 (기준 시드 {seed})", "p_py": "python / numpy",
        "split_h": "## 데이터와 나누기\n",
        "split_txt": ("에피소드 단위 나누기 (묶음 기준 `{group_by}`, 비율 맞춤 기준 `{strat}`, 시드 {seed}): "
                      "train {train}, val {val}, audit {audit}, 제외(사용 불가) {excluded}. "
                      "모든 조건과 반복 실행이 같은 audit 에피소드를 씁니다.\n"),
        "glob_h": "## 조건과 무관한 audit 결과\n", "q_v_s": "| 항목 | 값 | 출처 |",
        "vlm_acc": "audit 키프레임에서 VLM 단독 정확도",
        "vlm_acc_val": "{acc} (FPR {fpr}, FNR {fnr}, n={n}, VLM 제외 {vx}개, audit 제외 {ax}개)",
        "mock_vlm": "가짜 VLM (시뮬레이션)", "kappa": "블라인드 audit 재라벨링 일치도 (Cohen's kappa, 라벨 4종)",
        "kappa_val": "{k} (일치율 {o}, 키프레임 {n}개, 에피소드 {e}개)", "audit_timing": "audit 시간 출처: {t}",
        "audit_counts": "audit 라벨 개수", "reviewers": "검토자 {r}",
        "truth_row": "audit 라벨과 합성 정답 비교 (정확도)", "truth_src": "합성 데이터에서만 가능: 실제 데이터에는 정답이 없음",
        "cond_h": "## 조건별 결과 (반복 실행 평균 ± 표준편차, 아래 '불확실성 읽는 법' 참고)\n",
        "thr_txt": ("기본 임계값: `{thr}` (미리 정함, audit으로 튜닝하지 않음). 가짜 성공 종료 재현 설정: "
                    "연속 횟수 debounce_k={dk}, 허용 오차 {tol} 키프레임.\n"),
        "cost_h": "### 라벨링 비용 (사람 검토 시간 출처: **{src}**)\n",
        "cost_txt": ("'자른 시간'은 에피소드마다 최대 {cap:.0f}초로 자른 값입니다(자리를 비운 시간 대비). 두 값을 모두 보여 줍니다. "
                     "에피소드 수 예산이 같아도 시간 예산은 같지 않으니 시간 열을 비교하세요.\n"),
        "cost_cols": ("| 조건 | 예산 단위 | 검토한 에피소드 (k / N) | 검토한 키프레임 비율 | "
                      "사용한 시간 비율 (자른 시간 기준) | 사람 시간 [분] | 자른 사람 시간 [분] | 수정 횟수 | VLM 호출 수 |"),
        "unit_episodes": "에피소드", "unit_seconds": "시간",
        "lq_h": "### 학습 라벨 품질 (라벨링 풀 기준)\n",
        "lq_txt": ("기준 = 100% 사람 검토 라벨 (이것도 완벽하지는 않음). 애매함·무효·라벨 없음 키프레임은 학습에서 빼고 "
                   "개수를 셉니다.\n"),
        "lq_cols": "| 조건 | 100% 검토 대비 정확도 | FPR (가짜 성공) | FNR | 제외된 키프레임 |",
        "lq_truth_col": " 합성 정답 대비 정확도 |", "reference": "(기준)",
        "clf_h": "### 블라인드 audit 세트에서 보상 분류기 성능\n",
        "clf_src": "분류기 출처: **{src}**. audit 세트: 에피소드 {n}개.",
        "unc_txt": ("**불확실성 읽는 법.** `±`는 반복 실행 간 표준편차입니다. audit 세트가 고정이라 학습·선택의 "
                    "무작위성만 반영합니다. `[아래, 위]`는 에피소드 단위 부트스트랩 95% 구간(종료 비율은 Wilson 구간)을 "
                    "반복 실행에 걸쳐 평균한 것입니다. audit 에피소드가 적다는 사실을 반영하며 보통 훨씬 넓습니다. "
                    "조건끼리는 `±` 값이 아니라 아래 짝지은 비교 표로 비교하세요."),
        "clf_cols": ("| 조건 | 키프레임 정확도 | FPR | FNR | 정밀도 | 가짜 성공 종료 비율 | "
                     "놓친 성공 비율 | 점수 없음 |"),
        "paired_h": "#### `{ref}` 대비 짝지은 차이 (같은 audit 에피소드, 에피소드 짝 부트스트랩)\n",
        "paired_txt": ("값: 차이 (조건 − 기준), 반복 실행 평균, 반복 실행에 걸쳐 평균한 95% 구간. "
                       "구간에 0이 들어 있으면 이 audit 세트로는 차이를 구분하지 못했다는 뜻입니다.\n"),
        "paired_cols": "| 조건 | Δ 키프레임 정확도 | Δ FPR | Δ FNR | Δ 가짜 성공 종료 비율 |",
        "proxy_txt": ("가짜 성공 종료 비율은 오프라인 **추정치**입니다. 녹화된 audit 에피소드에 키프레임 간격으로 분류기 "
                      "점수를 다시 매긴 것이고, 실제 실행 주기보다 거칩니다. 온라인에서 잰 값이 아닙니다."),
        "no_scores": "**없음(MISSING)**: 이번 실행에는 분류기 점수가 제공되지 않았습니다.",
        "mp_h": "## 지표별 출처\n", "mp_cols": "| 지표 묶음 | 출처 |",
        "mp_data": "데이터", "mp_vlm": "VLM 제안",
        "mp_vlm_val": "모델 {m}, 지연 시간 {l}, 확신 점수 {c}",
        "mp_pool": "풀 검토 시간", "mp_pool_val": "{t} (검토자 {r})", "mp_audit": "audit 라벨링 시간",
        "mp_clf": "보상 분류기", "mp_robot": "| 실제 로봇 지표 | **없음(MISSING)** (아래 참고) |",
        "miss_h": "## 아직 없는 지표 (추정하거나 채워 넣지 않음)\n", "miss_item": "- **{name}**: 없음(MISSING). {why}.",
        "missing": [
            ("실제 로봇 과제 성공률 (사람이 판단)", "블라인드로 진행하는 실제 로봇 평가 에피소드가 필요합니다"),
            ("온라인 가짜 성공 종료 비율", "분류기가 끝낸 에피소드마다 사람이 판단한 결과가 필요합니다"),
            ("온라인 강화학습 중 사람 개입 횟수·시간", "실제 실행의 LeRobot actor 기록이 필요합니다"),
            ("학습 시간, 전체 소요 시간, 실행 실패율", "실제 HIL-SERL 실행이 필요합니다"),
            ("학습 시간당 검토한 라벨 수", "실제 학습 소요 시간이 필요합니다"),
            ("로봇 성공률의 시드 간 분산", "조건마다 실제 시드가 2개 이상 필요합니다"),
        ],
        "guard_h": "## 해석할 때 주의할 점\n",
        "guard_reps": "- 조건마다 반복 실행 {n}회. 차이는 기술(묘사)일 뿐이며 유의성 검정은 하지 않았습니다.",
        "guard_robot": ("- 라벨이나 분류기 정확도가 좋아져도 로봇 성공률이 좋아진다는 뜻은 아닙니다. 그것은 블라인드 실제 로봇 "
                        "평가로만 보일 수 있습니다."),
        "guard_syn": ("- 여기서 조건 간 차이는 시뮬레이터에 넣은 지어낸 오류·시간 수치에서 나온 것입니다. 그 수치를 바꾸면 "
                      "결론도 바뀝니다."),
        "undefined": "정의 안 됨", "none": "없음",
        "values": {"measured": "실측", "simulated": "시뮬레이션", "missing": "없음", "not_applicable": "해당 없음",
                   "mixed": "혼합", "synthetic": "합성", "real": "실제", "mock": "가짜", "logprob": "logprob",
                   "self_reported": "자기 보고", "none": "없음"},
    },
}


def _tr(T: Mapping[str, Any], value: Any) -> str:
    """Translate a known source/kind token (e.g. 'simulated' -> '시뮬레이션'); other values unchanged."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_tr(T, v) for v in value) + "]"
    return str(T["values"].get(value, value)) if isinstance(value, str) else str(value)


def _classifier_source(T: Mapping[str, Any], lang: str, src: str | None) -> str:
    if lang != "ko" or not src:
        return str(src)
    if src.startswith("logistic stand-in"):
        return "합성 특징에 대한 로지스틱 회귀 대체 분류기 (시뮬레이션)"
    if src.startswith("scores file:"):
        return "점수 파일:" + src[len("scores file:"):]
    if src.startswith("MISSING"):
        return "없음(MISSING) (분류기 점수가 제공되지 않음)"
    return src


def _evidence(lang: str, status: str | None) -> str:
    if lang == "ko" and status:
        if status.startswith("SYNTHETIC"):
            return "합성: 프로그램 점검용, 실제 시스템에 대한 증거 아님"
        if status == "REAL DATA":
            return "실제 데이터"
    return str(status)


def _fmt(stat: Mapping[str, Any] | None, digits: int = 3, scale: float = 1.0, undefined: str = "undefined") -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    m = stat["mean"] * scale
    sd = stat.get("sd")
    n = stat.get("n", 0)
    s = f"{m:.{digits}f}"
    if sd is not None:
        s += f" ± {sd * scale:.{digits}f}"
    if stat.get("n_missing"):
        s += f" ({stat['n_missing']} {undefined})"
    return s if n else "n/a"


def _get(summary_cond: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    return summary_cond.get(key)


def _ci(summary_cond: Mapping[str, Any], prefix: str, digits: int = 3) -> str:
    """Replicate-averaged [lo, hi] for an interval stored as {prefix}.lo / {prefix}.hi; '[n/a]' if unavailable."""
    lo, hi = summary_cond.get(f"{prefix}.lo"), summary_cond.get(f"{prefix}.hi")
    if not lo or not hi or lo.get("mean") is None or hi.get("mean") is None:
        return "[n/a]"
    return f"[{lo['mean']:.{digits}f}, {hi['mean']:.{digits}f}]"


def _plain(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_report(provenance: Mapping[str, Any], summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    lang = cfg.get("report_lang", "ko")
    if lang not in TEXT:
        raise ValueError(f"report_lang must be one of {sorted(TEXT)}, got {lang!r}")
    T = TEXT[lang]
    synthetic = provenance.get("data_kind") == "synthetic"
    src = summary.get("sources", {})
    timing_src = _tr(T, "/".join(src.get("pool_review_timing_sources", [])) or "none")
    audit_timing = _tr(T, "/".join(src.get("audit_timing_sources", [])) or "none")
    vlm_src = T["mp_vlm_val"].format(m=src.get("vlm_models"), l=_tr(T, src.get("vlm_latency_sources", [])),
                                     c=_tr(T, src.get("vlm_confidence_sources", [])))
    vlm_short = T["mock_vlm"] if "mock-vlm" in src.get("vlm_models", []) else f"VLM {src.get('vlm_models')}"
    clf_src = _classifier_source(T, lang, src.get("classifier"))
    und = T["undefined"]
    lines: list[str] = [T["title"].format(name=provenance["experiment_name"])]
    if synthetic:
        lines.append(T["banner"])
    git = provenance.get("git", {})
    lines += [
        T["prov_h"], T["field_value"], "|---|---|",
        f"| {T['p_kind']} | **{_tr(T, provenance['data_kind'])}** ({_evidence(lang, provenance.get('evidence_status'))}) |",
        f"| {T['p_hash']} | `{provenance['config_hash']}` |",
        f"| {T['p_git']} | `{git.get('commit')}` (dirty={git.get('dirty')}) |",
        f"| {T['p_created']} | {provenance['created_at_utc']} |",
        f"| {T['p_reps']} | {T['p_reps_val'].format(n=provenance['replicates'], seed=provenance['base_seed'])} |",
        f"| {T['p_py']} | {provenance['environment']['python']} / {provenance['environment']['numpy']} |",
        "",
    ]
    sc = summary["split_counts"]
    lines += [
        T["split_h"],
        T["split_txt"].format(group_by=cfg["splits"]["group_by"], strat=cfg["splits"]["stratify_by"],
                              seed=cfg["splits"]["seed"], **sc),
    ]
    g = summary["global"]
    vz = g["vlm_zero_shot_vs_audit"]
    ag = g["audit_repeat_agreement"]
    lines += [
        T["glob_h"], T["q_v_s"], "|---|---|---|",
        f"| {T['vlm_acc']} | " + T["vlm_acc_val"].format(
            acc=_plain(vz.get("accuracy")), fpr=_plain(vz.get("fpr")), fnr=_plain(vz.get("fnr")), n=vz.get("n"),
            vx=vz.get("n_vlm_excluded"), ax=vz.get("n_ref_excluded")) + f" | {vlm_short} |",
        f"| {T['kappa']} | " + T["kappa_val"].format(k=_plain(ag.get("kappa")), o=_plain(ag.get("observed_agreement")),
                                                     n=ag.get("n"), e=ag.get("n_episodes"))
        + f" | {T['audit_timing'].format(t=audit_timing)} |",
        f"| {T['audit_counts']} | {g['audit_label_counts']} | {T['reviewers'].format(r=src.get('reviewer_ids'))} |",
    ]
    if "audit_labels_vs_synthetic_truth" in g:
        t = g["audit_labels_vs_synthetic_truth"]
        lines.append(f"| {T['truth_row']} | {_plain(t.get('accuracy'))} | {T['truth_src']} |")
    lines.append("")

    conds = summary["conditions"]
    lines += [
        T["cond_h"],
        T["thr_txt"].format(thr=cfg["threshold"], dk=cfg["termination"]["debounce_k"],
                            tol=cfg["termination"]["tolerance"]),
        T["cost_h"].format(src=timing_src),
        T["cost_txt"].format(cap=float(cfg["effort"]["idle_cap_s"])),
        T["cost_cols"], "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cfg["conditions"]:
        s = conds.get(c["name"], {})
        unit = T["unit_seconds"] if c.get("budget_unit", "episodes") == "seconds" else T["unit_episodes"]
        lines.append(
            f"| {c['name']} | {unit} | {_fmt(_get(s, 'budget.k'), 1, undefined=und)} / "
            f"{_fmt(_get(s, 'budget.n_eligible_pool_episodes'), 0, undefined=und)} | "
            f"{_fmt(_get(s, 'budget.realized_keyframe_fraction'), 3, undefined=und)} | "
            f"{_fmt(_get(s, 'budget.realized_time_fraction'), 3, undefined=und)} | "
            f"{_fmt(_get(s, 'effort.human_time_s'), 1, 1 / 60, und)} | "
            f"{_fmt(_get(s, 'effort.human_time_s_capped'), 1, 1 / 60, und)} | "
            f"{_fmt(_get(s, 'effort.n_corrections'), 1, undefined=und)} | "
            f"{_fmt(_get(s, 'vlm_cost.n_calls'), 0, undefined=und)} |"
        )
    lines += ["", T["lq_h"], T["lq_txt"],
              T["lq_cols"] + (T["lq_truth_col"] if synthetic else ""),
              "|---|---|---|---|---|" + ("---|" if synthetic else "")]
    for c in cfg["conditions"]:
        s = conds.get(c["name"], {})
        name = f"{c['name']} {T['reference']}" if c["policy"] == "full" else c["name"]
        row = (f"| {name} | {_fmt(_get(s, 'label_quality_vs_full_review.accuracy'), undefined=und)} | "
               f"{_fmt(_get(s, 'label_quality_vs_full_review.fpr'), undefined=und)} | "
               f"{_fmt(_get(s, 'label_quality_vs_full_review.fnr'), undefined=und)} | "
               f"{_fmt(_get(s, 'label_counts.n_excluded'), 1, undefined=und)} |")
        if synthetic:
            row += f" {_fmt(_get(s, 'label_quality_vs_synthetic_truth.accuracy'), undefined=und)} |"
        lines.append(row)
    has_audit = any("audit.keyframe.accuracy" in conds.get(c["name"], {}) for c in cfg["conditions"])
    lines += ["", T["clf_h"]]
    if has_audit:
        lines += [T["clf_src"].format(src=clf_src, n=summary["split_counts"]["audit"]), "", T["unc_txt"], "",
                  T["clf_cols"], "|---|---|---|---|---|---|---|---|"]
        for c in cfg["conditions"]:
            s = conds.get(c["name"], {})
            lines.append(
                f"| {c['name']} | {_fmt(_get(s, 'audit.keyframe.accuracy'), undefined=und)} "
                f"{_ci(s, 'audit.keyframe.ci95_episode_bootstrap.accuracy')} | "
                f"{_fmt(_get(s, 'audit.keyframe.fpr'), undefined=und)} {_ci(s, 'audit.keyframe.ci95_episode_bootstrap.fpr')} | "
                f"{_fmt(_get(s, 'audit.keyframe.fnr'), undefined=und)} {_ci(s, 'audit.keyframe.ci95_episode_bootstrap.fnr')} | "
                f"{_fmt(_get(s, 'audit.keyframe.precision'), undefined=und)} | "
                f"{_fmt(_get(s, 'audit.termination.fst_rate'), undefined=und)} {_ci(s, 'audit.termination.fst_rate_ci95')} | "
                f"{_fmt(_get(s, 'audit.termination.missed_success_rate'), undefined=und)} | "
                f"{_fmt(_get(s, 'audit.keyframe.n_score_missing'), 0, undefined=und)} |"
            )
        ref = next((c["name"] for c in cfg["conditions"] if c["policy"] == "full"), None)
        if ref is not None:
            lines += ["", T["paired_h"].format(ref=ref), T["paired_txt"], T["paired_cols"], "|---|---|---|---|---|"]
            for c in cfg["conditions"]:
                if c["name"] == ref:
                    continue
                s = conds.get(c["name"], {})
                cells = [f"{_fmt(_get(s, f'paired_vs_reference.{m}.diff'), undefined=und)} "
                         f"{_ci(s, f'paired_vs_reference.{m}')}" for m in ("accuracy", "fpr", "fnr", "fst_rate")]
                lines.append(f"| {c['name']} | " + " | ".join(cells) + " |")
        lines += ["", T["proxy_txt"], ""]
    else:
        lines += [T["no_scores"], ""]

    lines += [
        T["mp_h"], T["mp_cols"], "|---|---|",
        f"| {T['mp_data']} | {_tr(T, provenance['data_kind'])} |",
        f"| {T['mp_vlm']} | {vlm_src} |",
        f"| {T['mp_pool']} | {T['mp_pool_val'].format(t=timing_src, r=src.get('reviewer_ids'))} |",
        f"| {T['mp_audit']} | {audit_timing} |",
        f"| {T['mp_clf']} | {clf_src} |",
        T["mp_robot"], "", T["miss_h"],
    ]
    for name, why in T["missing"]:
        lines.append(T["miss_item"].format(name=name, why=why))
    lines += ["", T["guard_h"], T["guard_reps"].format(n=provenance["replicates"]), T["guard_robot"]]
    if synthetic:
        lines.append(T["guard_syn"])
    lines.append("")
    return "\n".join(lines)
